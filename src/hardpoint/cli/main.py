"""The ``hardpoint`` command.

Thin by design: each command loads configuration, builds resources, calls the
library, and prints. Anything worth testing lives in ``cli/commands/`` or in the
library, so the CLI never becomes a second implementation of what it drives.

Every ``HardpointError`` is printed with its remedy and exits non-zero -- 2 for
configuration problems, 1 for everything else -- rather than as a traceback.
"""

import json
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import anyio
import typer
import yaml

import hardpoint
from hardpoint.cli.commands.ask import ask as run_ask
from hardpoint.cli.commands.ask import render_answer, render_explain
from hardpoint.cli.commands.doctor import (
    component_checks,
    configured_components,
    render_checks,
    run_doctor,
)
from hardpoint.cli.commands.evaluate import run_suite
from hardpoint.cli.commands.ingest import ingest as run_ingest
from hardpoint.cli.commands.ingest import ingest_status
from hardpoint.cli.commands.init import TEMPLATES, diff_project, write_project
from hardpoint.cli.commands.project import CliState, load_project_config, project_environ
from hardpoint.cli.commands.schema import config_schema
from hardpoint.core.config.snapshot import flatten
from hardpoint.core.errors import ConfigError, HardpointError, MissingDependencyError
from hardpoint.core.registry import ComponentRegistry, Kind
from hardpoint.runtime.policies import PolicyChain
from hardpoint.runtime.resources import Resources, build_resources

__all__ = ["app"]

T = TypeVar("T")

app = typer.Typer(
    name="hardpoint",
    help="Contracts, runtime, ingestion and evaluation for production RAG and agentic systems.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Inspect and validate configuration.", no_args_is_help=True)
components_app = typer.Typer(help="Inspect registered components.", no_args_is_help=True)
ingest_app = typer.Typer(help="Sync sources into the index.", no_args_is_help=True)
eval_app = typer.Typer(help="Run evaluation suites and quality gates.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(components_app, name="components")
app.add_typer(ingest_app, name="ingest")
app.add_typer(eval_app, name="eval")


# --------------------------------------------------------------------------- #
# Plumbing                                                                    #
# --------------------------------------------------------------------------- #


def _state(ctx: typer.Context) -> CliState:
    state = ctx.obj
    return state if isinstance(state, CliState) else CliState()


def _fail(exc: HardpointError) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(2 if isinstance(exc, ConfigError) else 1)


def _run(work: Callable[[], Awaitable[T]]) -> T:
    """Run async work, turning a ``HardpointError`` into its message and remedy."""
    try:
        return anyio.run(work)
    except HardpointError as exc:
        raise _fail(exc) from None


async def _with_resources(state: CliState, work: Callable[[Resources], Awaitable[T]]) -> T:
    res = await build_resources(load_project_config(state))
    try:
        return await work(res)
    finally:
        await res.aclose()


@app.callback()
def main(
    ctx: typer.Context,
    project_dir: Annotated[
        Path, typer.Option("--project-dir", "-C", help="Run as if started in this directory.")
    ] = Path(),
    env: Annotated[
        str | None, typer.Option("--env", help="Configuration environment (HARDPOINT_ENV).")
    ] = None,
    config_dir: Annotated[
        str, typer.Option("--config-dir", help="Directory holding base.yaml and overlays.")
    ] = "config",
) -> None:
    """Hardpoint: build RAG and agentic systems that survive production."""
    if project_dir != Path():
        if not project_dir.is_dir():
            typer.echo(f"error: {project_dir} is not a directory", err=True)
            raise typer.Exit(2)
        os.chdir(project_dir)
    ctx.obj = CliState(env=env, config_dir=config_dir)


@app.command()
def version() -> None:
    """Print the installed version and the port contract version."""
    typer.echo(f"hardpoint {hardpoint.__version__} (contract {hardpoint.CONTRACT_VERSION})")


# --------------------------------------------------------------------------- #
# init                                                                        #
# --------------------------------------------------------------------------- #


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Directory to create the project in.")],
    template: Annotated[
        str, typer.Option("--template", "-t", help=f"One of: {', '.join(TEMPLATES)}.")
    ] = "rag-minimal",
    diff: Annotated[
        bool,
        typer.Option("--diff", help="Show what a fresh template would change. Writes nothing."),
    ] = False,
) -> None:
    """Generate a project from a template."""
    destination = Path(name)
    try:
        if diff:
            typer.echo(diff_project(destination, template) or "no differences")
            return
        written = write_project(destination, template)
    except HardpointError as exc:
        raise _fail(exc) from None
    typer.echo(f"created {destination} from {template!r} ({len(written)} files)")
    typer.echo("")
    typer.echo(f"  cd {destination}")
    typer.echo("  cp .env.example .env      # add your keys, or use --env offline")
    typer.echo("  hardpoint doctor")
    typer.echo("  hardpoint ingest run --plan")
    typer.echo("  hardpoint ingest run")
    typer.echo('  hardpoint ask "How do I rotate an API key?" --explain')


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #


@config_app.command("show")
def config_show(
    ctx: typer.Context,
    as_yaml: Annotated[
        bool, typer.Option("--yaml", help="Print as YAML, without origins.")
    ] = False,
) -> None:
    """Print the resolved configuration, redacted, with each key's origin layer."""
    try:
        resolved = load_project_config(_state(ctx))
    except HardpointError as exc:
        raise _fail(exc) from None
    snapshot = resolved.snapshot
    data = snapshot.to_dict()
    if as_yaml:
        typer.echo(yaml.safe_dump(data, sort_keys=True).rstrip())
        return
    typer.echo(f"# env {snapshot.env}, hash {snapshot.hash[:12]}")
    for path, value in flatten(data):
        typer.echo(f"{path} = {json.dumps(value)}    # {snapshot.origin(path).value}")


@config_app.command("schema")
def config_schema_command() -> None:
    """Print the JSON Schema for configuration, including every registered component."""
    try:
        typer.echo(json.dumps(config_schema(ComponentRegistry()), indent=2, sort_keys=True))
    except HardpointError as exc:
        raise _fail(exc) from None


@config_app.command("validate")
def config_validate(ctx: typer.Context) -> None:
    """Validate configuration and every component's options, without connecting to anything."""
    try:
        resolved = load_project_config(_state(ctx))
        checks = component_checks(resolved.config, ComponentRegistry())
        failures = [check for check in checks if check.status == "fail"]
    except HardpointError as exc:
        raise _fail(exc) from None
    if failures:
        for check in failures:
            typer.echo(f"error: {check.name}: {check.detail}\n  fix: {check.remedy}", err=True)
        raise typer.Exit(2)
    typer.echo(f"ok: env {resolved.snapshot.env!r}, hash {resolved.snapshot.hash[:12]}")


# --------------------------------------------------------------------------- #
# components                                                                  #
# --------------------------------------------------------------------------- #


@components_app.command("list")
def components_list(
    ctx: typer.Context,
    kind: Annotated[str | None, typer.Option("--type", help="Only this kind.")] = None,
    describe: Annotated[
        bool, typer.Option("--describe", help="Print each component's options.")
    ] = False,
    resolved: Annotated[
        bool, typer.Option("--resolved", help="Print configured components and their policies.")
    ] = False,
) -> None:
    """List registered components: key, kind, where it came from, and its extra."""
    registry = ComponentRegistry()
    if resolved:
        try:
            config = load_project_config(_state(ctx)).config
        except HardpointError as exc:
            raise _fail(exc) from None
        for spec_kind, path, spec in configured_components(config):
            chain = PolicyChain.from_config(spec.policies).describe()
            fallback = (
                f" (fallback: {spec.policies.fallback.type})" if spec.policies.fallback else ""
            )
            typer.echo(f"{path:<32} {spec_kind.value:<11} {spec.type:<20} {chain}{fallback}")
        return

    try:
        kinds = [Kind(kind)] if kind else list(Kind)
    except ValueError:
        typer.echo(
            f"error: unknown kind {kind!r}; one of {', '.join(k.value for k in Kind)}", err=True
        )
        raise typer.Exit(2) from None

    for each in kinds:
        for key, source, extra in registry.describe(each):
            typer.echo(f"{each.value:<11} {key:<20} {source:<10} {extra or '-'}")
            if describe:
                try:
                    model = registry.resolve(each, key).config_model
                except HardpointError as exc:
                    typer.echo(f"    (unavailable: {exc.message})")
                    continue
                for field_name, field in model.model_fields.items():
                    default = "required" if field.is_required() else repr(field.default)
                    typer.echo(f"    {field_name}: {field.annotation!s}  [{default}]")


# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #


SourceOption = Annotated[
    list[str] | None, typer.Option("--source", "-s", help="Only this source. Repeatable.")
]


@ingest_app.command("run")
def ingest_run(
    ctx: typer.Context,
    source: SourceOption = None,
    plan: Annotated[
        bool, typer.Option("--plan", help="Show the diff and cost; change nothing.")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Abort on the first failure.")
    ] = False,
) -> None:
    """Bring the index into line with the sources, doing as little work as possible."""
    state = _state(ctx)
    text, code = _run(
        lambda: _with_resources(
            state, lambda res: run_ingest(res, source or [], plan=plan, fail_fast=fail_fast)
        )
    )
    typer.echo(text)
    if code:
        raise typer.Exit(code)


@ingest_app.command("plan")
def ingest_plan(ctx: typer.Context, source: SourceOption = None) -> None:
    """Show what ``ingest run`` would do and what it would cost. Changes nothing."""
    state = _state(ctx)
    text, _ = _run(
        lambda: _with_resources(state, lambda res: run_ingest(res, source or [], plan=True))
    )
    typer.echo(text)


@ingest_app.command("status")
def ingest_status_command(ctx: typer.Context, source: SourceOption = None) -> None:
    """Show each source's last run and each index's epoch."""
    state = _state(ctx)
    typer.echo(_run(lambda: _with_resources(state, lambda res: ingest_status(res, source or []))))


# --------------------------------------------------------------------------- #
# ask                                                                         #
# --------------------------------------------------------------------------- #


@app.command("ask")
def ask_command(
    ctx: typer.Context,
    question: Annotated[str, typer.Argument(help="The question.")],
    explain: Annotated[
        bool, typer.Option("--explain", help="Print the execution report after the answer.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the Answer as JSON.")] = False,
) -> None:
    """Answer one question through the project's pipeline."""
    state = _state(ctx)

    async def work(res: Resources) -> tuple[Any, ...]:
        answer, captured = await run_ask(res, question)
        redact = "messages.content" in res.config.observability.redact
        return answer, captured, redact

    answer, captured, redact = _run(lambda: _with_resources(state, work))
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
        return
    typer.echo(render_answer(answer))
    if explain:
        typer.echo("")
        typer.echo(render_explain(answer, captured, redact_content=redact))


# --------------------------------------------------------------------------- #
# eval                                                                        #
# --------------------------------------------------------------------------- #


@eval_app.command("run")
def eval_run(  # noqa: PLR0917 - one parameter per command-line option
    ctx: typer.Context,
    suite: Annotated[
        str, typer.Option("--suite", help="Dataset: <datasets_dir>/<suite>.jsonl.")
    ] = "smoke",
    tag: Annotated[
        list[str] | None, typer.Option("--tag", help="Only cases with this tag.")
    ] = None,
    max_cost: Annotated[
        float | None, typer.Option("--max-cost", help="Refuse to start above this estimate (USD).")
    ] = None,
    cassettes: Annotated[
        str, typer.Option("--cassettes", help="off, record, or replay (zero provider calls).")
    ] = "off",
    update_baseline: Annotated[
        bool, typer.Option("--update-baseline", help="Write this run as the committed baseline.")
    ] = False,
    output: Annotated[Path, typer.Option("--output", help="Where reports are written.")] = Path(
        "artefacts/eval"
    ),
) -> None:
    """Run an eval suite through the project's pipeline and gate it."""
    if cassettes not in ("off", "record", "replay"):
        typer.echo("error: --cassettes must be off, record or replay", err=True)
        raise typer.Exit(2)
    state = _state(ctx)
    report, markdown = _run(
        lambda: _with_resources(
            state,
            lambda res: run_suite(
                res,
                suite,
                tags=tuple(tag or ()),
                max_cost=max_cost,
                cassettes=cassettes,  # type: ignore[arg-type]  # validated above
                update_baseline=update_baseline,
                output=output,
            ),
        )
    )
    typer.echo(markdown)
    if report.gate is not None and not report.gate.passed:
        raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# serve                                                                       #
# --------------------------------------------------------------------------- #


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = 1,
) -> None:
    """Run the project's HTTP service (``project.service``) under uvicorn.

    In-flight requests get the request deadline (``budgets.request.deadline_s``)
    to finish on shutdown before the service closes its connections.
    """
    state = _state(ctx)
    try:
        resolved = load_project_config(state)
        uvicorn = _import_uvicorn()
    except HardpointError as exc:
        raise _fail(exc) from None

    # The server process reads configuration itself; hand it the project's
    # .env and selected environment the same way the other commands see them.
    for key, value in project_environ().items():
        os.environ.setdefault(key, value)
    if state.env:
        os.environ["HARDPOINT_ENV"] = state.env

    target = resolved.config.project.service
    module, _, factory = target.partition(":")
    grace = resolved.config.budgets.request.deadline_s or 30
    sys.path.insert(0, str(Path.cwd()))
    uvicorn.run(
        f"{module}:{factory}",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        timeout_graceful_shutdown=int(grace),
    )


def _import_uvicorn() -> Any:
    try:
        import uvicorn  # noqa: PLC0415 - the serve extra, only for this command
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "`hardpoint serve` requires the 'serve' extra, which is not installed.",
            extra="serve",
            component="serve",
            remedy="pip install 'hardpoint[serve]'",
            cause=exc,
        ) from exc
    return uvicorn


# --------------------------------------------------------------------------- #
# doctor                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def doctor(
    ctx: typer.Context,
    live: Annotated[
        bool, typer.Option("--live", help="Also make one tiny call per provider.")
    ] = False,
) -> None:
    """Check configuration, environment, extras, connectivity and dimensions."""
    state = _state(ctx)
    checks = _run(lambda: run_doctor(state, live=live))
    typer.echo(render_checks(checks))
    if any(check.status == "fail" for check in checks):
        raise typer.Exit(1)
