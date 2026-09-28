"""Configured components, and the one path from a question to an ``Answer``.

``build_resources`` turns a resolved configuration into live components through
the registry, wraps each port in its configured policies, and hands back a
:class:`Resources`. A project's pipeline factory receives it::

    def build(res: Resources) -> Pipeline:
        return Pipeline("rag", [VectorRetriever(res.index(), res.embedder), ...])

## One factory, three callers

The CLI's ``ask``, the generated service and the eval runner all build through
:func:`load_pipeline` and answer through :func:`answer_query`. There is no
second execution path for evaluation (INSTRUCTIONS.md §13.10, ADR-011): if the
service would behave differently from what the eval suite measured, the suite
would be measuring a different system.

## Paths are relative to the working directory

Which is the project directory: the CLI changes to it, the service runs from
it. Components receive their configured paths unchanged.
"""

from __future__ import annotations

import importlib
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from hardpoint.core.config.snapshot import ConfigSnapshot
from hardpoint.core.context import Budget, CacheHandle, Deadline, RunContext, UsageAccumulator
from hardpoint.core.errors import ConfigError, ContractError
from hardpoint.core.models import Answer
from hardpoint.core.registry import ComponentRegistry, Kind
from hardpoint.observability.metrics import NoOpMetricSink
from hardpoint.observability.pricing import PricingTable
from hardpoint.observability.tracing import NoOpTracer, RedactingTracer
from hardpoint.runtime.policies import PolicyChain
from hardpoint.runtime.wrapped import (
    PolicyEmbeddingModel,
    PolicyLanguageModel,
    PolicyReranker,
    PolicyVectorIndex,
)

if TYPE_CHECKING:
    from hardpoint.core.config.loader import ResolvedConfig
    from hardpoint.core.config.schema import ComponentSpec, GuardSpec, HardpointConfig
    from hardpoint.core.ports import (
        CacheBackend,
        EmbeddingModel,
        LanguageModel,
        MetricSink,
        PromptStore,
        Reranker,
        StateStore,
        Tracer,
        VectorIndex,
    )
    from hardpoint.generation.generate import Generate
    from hardpoint.guards.base import GuardCheck
    from hardpoint.retrieval.retrievers import Assembled, EpochReader
    from hardpoint.runtime.pipeline import DeltaSink, Pipeline
    from hardpoint.runtime.step import Step

T = TypeVar("T")

__all__ = [
    "Resources",
    "answer_query",
    "build_resources",
    "build_source",
    "load_pipeline",
    "new_run_id",
]


def new_run_id() -> str:
    """Return a fresh run identifier.

    Random on purpose: a run id must be unique, not stable. Stable identity is
    for documents and chunks (``core.ids``), never for runs.
    """
    return f"run_{uuid.uuid4().hex[:20]}"


@dataclass(frozen=True)
class Resources:
    """Everything a pipeline factory can build from.

    Components that were not configured are ``None`` in their slot, and the
    accessor raises a ``ConfigError`` naming the config path, so a pipeline that
    needs an LLM the project never configured fails at construction with a
    remedy rather than with ``AttributeError: 'NoneType'``.

    Args:
        config: The validated configuration.
        snapshot: The hashed, redacted snapshot of it.
        registry: The registry components were resolved through.
        llm_: The language model, policy-wrapped.
        embedder_: The embedding model, policy-wrapped.
        reranker_: The reranker, policy-wrapped.
        indexes: Named vector indexes, policy-wrapped.
        state: The ingestion manifest.
        prompts: The project's prompt store.
        tracer: Where spans go, already wrapped in the configured redaction.
        metrics: Where measurements go.
        shared_cache: The backend behind ``cache.backend``, or ``None``.
        pricing: Model prices, shipped and overridden.
        input_checks: The checks configured under ``guards.input``.
        output_checks: The checks configured under ``guards.output``.
    """

    config: HardpointConfig
    snapshot: ConfigSnapshot
    registry: ComponentRegistry
    llm_: LanguageModel | None
    embedder_: EmbeddingModel | None
    reranker_: Reranker | None
    indexes: Mapping[str, VectorIndex]
    state: StateStore
    prompts: PromptStore
    tracer: Tracer = field(default_factory=NoOpTracer)
    metrics: MetricSink = field(default_factory=NoOpMetricSink)
    shared_cache: CacheBackend | None = None
    pricing: PricingTable = field(default_factory=PricingTable)
    input_checks: tuple[GuardCheck[str], ...] = ()
    output_checks: tuple[GuardCheck[Answer], ...] = ()

    @property
    def llm(self) -> LanguageModel:
        """The configured language model.

        Raises:
            ConfigError: If ``providers.llm`` is not configured.
        """
        return _require(self.llm_, "providers.llm", "a language model")

    @property
    def embedder(self) -> EmbeddingModel:
        """The configured embedding model.

        Raises:
            ConfigError: If ``providers.embeddings`` is not configured.
        """
        return _require(self.embedder_, "providers.embeddings", "an embedding model")

    @property
    def reranker(self) -> Reranker:
        """The configured reranker.

        Raises:
            ConfigError: If ``providers.reranker`` is not configured.
        """
        return _require(self.reranker_, "providers.reranker", "a reranker")

    def index(self, name: str = "primary") -> VectorIndex:
        """Return a named index.

        Raises:
            ConfigError: If no index of that name is configured.
        """
        found = self.indexes.get(name)
        if found is None:
            raise ConfigError(
                f"No index named {name!r} is configured.",
                config_path=f"indexes.{name}",
                remedy=(
                    f"Configured indexes: {', '.join(sorted(self.indexes)) or '(none)'}. "
                    f"Add `indexes.{name}` with a `type:` such as `sqlite` or `qdrant`."
                ),
            )
        return found

    def input_guards(self) -> list[Step[str, Any]]:
        """The configured input guards as one step, or no step: ``[*res.input_guards(), ...]``."""
        from hardpoint.guards.base import InputGuard  # noqa: PLC0415 - guards import runtime

        return [InputGuard(self.input_checks)] if self.input_checks else []

    def guarded(self, generate: Generate, *, refusal: str | None = None) -> Step[Assembled, Any]:
        """Wrap generation in the configured output guards, when there are any.

        With output guards the answer is checked before anyone sees it, so a
        streamed request receives it whole: streaming text that a guard then
        withdraws would show the user what the guard exists to withhold.
        """
        if not self.output_checks:
            return generate
        from hardpoint.guards.base import GuardedGenerate  # noqa: PLC0415 - guards import runtime

        return GuardedGenerate(generate, self.output_checks, refusal=refusal, name=generate.name)

    def with_components(self, **changes: Any) -> Resources:
        """Return a copy with some components replaced.

        How the CLI wraps the model to capture the rendered prompt for
        ``--explain``, and how the eval runner puts cassettes in front of
        providers -- by substitution, visibly, rather than through a hook.
        """
        return replace(self, **changes)

    def run_context(self, *, run_id: str | None = None) -> RunContext:
        """Build a ``RunContext``: the configured budget and deadline, and caches.

        Each run gets a fresh request-scoped cache and shares the configured
        backend.
        """
        budget_config = self.config.budgets.request
        return RunContext(
            run_id=run_id or new_run_id(),
            tracer=self.tracer,
            metrics=self.metrics,
            deadline=Deadline.in_seconds(budget_config.deadline_s),
            budget=Budget.from_config(budget_config),
            cache=CacheHandle(request=_RequestCache(), shared=self.shared_cache),
            config=self.snapshot,
            usage=UsageAccumulator(),
            extras={},
        )

    def epoch_reader(self, name: str = "primary") -> EpochReader:
        """Return a callable reading an index's current epoch from the manifest.

        What ``VectorRetriever(epoch=...)`` takes to key its cache on the epoch,
        so an ingestion run invalidates retrievals built on the old index.
        """

        async def read() -> int:
            await self.state.initialise()
            return await self.state.index_epoch(name)

        return read

    async def epochs(self) -> dict[str, int]:
        """Return each configured index's epoch from the manifest.

        The manifest owns the epoch (ADR-009), so it is read there rather than
        from the index.
        """
        await self.state.initialise()
        return {name: await self.state.index_epoch(name) for name in sorted(self.indexes)}

    async def aclose(self) -> None:
        """Close every component that holds a connection, a file or a buffer."""
        seen: set[int] = set()
        for component in (
            self.llm_,
            self.embedder_,
            self.reranker_,
            *self.indexes.values(),
            self.shared_cache,
            self.metrics,
            getattr(self.tracer, "inner", self.tracer),
        ):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            closer = getattr(component, "aclose", None)
            if closer is not None:
                await closer()
        close_state = getattr(self.state, "close", None)
        if close_state is not None:
            await close_state()


class _RequestCache:
    """The request-scoped cache: a dict that lives and dies with one run."""

    def __init__(self) -> None:
        self._values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._values.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int | None) -> None:  # noqa: ARG002 - dies with the run
        self._values[key] = value

    async def delete_prefix(self, prefix: str) -> int:
        doomed = [key for key in self._values if key.startswith(prefix)]
        for key in doomed:
            del self._values[key]
        return len(doomed)


def _require(component: T | None, path: str, what: str) -> T:
    if component is None:
        raise ConfigError(
            f"The pipeline needs {what}, and none is configured.",
            config_path=path,
            remedy=f"Add `{path}` with a `type:`. `hardpoint components list` shows the options.",
        )
    return component


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


async def _create(
    registry: ComponentRegistry,
    kind: Kind,
    spec: ComponentSpec,
    path: str,
    defaults: Mapping[str, Any] | None = None,
) -> Any:
    """Construct one component, filling defaults the component's own model declares.

    ``defaults`` carries values only the caller knows -- an index's configured
    name -- and is applied only to fields the component's config model has and
    the user did not set.
    """
    registration = registry.resolve(kind, spec.type, config_path=f"{path}.type")
    options = dict(spec.options())
    for key, value in (defaults or {}).items():
        if key in registration.config_model.model_fields and key not in options:
            options[key] = value
    return await registry.create(kind, spec.type, options, config_path=path)


async def _create_with_policies(
    registry: ComponentRegistry,
    kind: Kind,
    spec: ComponentSpec,
    path: str,
    *,
    config: HardpointConfig,
    pricing: PricingTable,
) -> Any:
    """Construct a component and its fallback, wrapped in policies, tracing and cost."""
    inner = await _create(registry, kind, spec, path)
    chain = PolicyChain.from_config(spec.policies)
    fallback_spec = spec.policies.fallback
    fallback = (
        await _create(registry, kind, fallback_spec, f"{path}.policies.fallback")
        if fallback_spec is not None
        else None
    )

    if kind is Kind.LLM:
        return PolicyLanguageModel(inner, chain, fallback, pricing=pricing)
    if kind is Kind.EMBEDDINGS:
        return PolicyEmbeddingModel(
            inner, chain, fallback, pricing=pricing, cache=config.cache.embeddings
        )
    return PolicyReranker(inner, chain, fallback, pricing=pricing, cache=config.cache.rerank)


async def build_resources(
    resolved: ResolvedConfig,
    *,
    registry: ComponentRegistry | None = None,
    prompts: PromptStore | None = None,
) -> Resources:
    """Construct every configured component.

    Args:
        resolved: The loaded configuration.
        registry: Where components are resolved. Defaults to a fresh registry
            with the built-in table, plus entry-point discovery when
            ``plugins.discover`` is on.
        prompts: The prompt store. Defaults to the files in
            ``project.prompts_dir``, or an empty store when that directory does
            not exist.

    Returns:
        The resources, ready for a pipeline factory.

    Raises:
        UnknownComponentError: For a ``type:`` nothing is registered under.
        MissingDependencyError: For a component whose extra is not installed.
        InvalidConfigError: For invalid component options.
    """
    config = resolved.config
    registry = registry or ComponentRegistry()
    if config.plugins.discover:
        from importlib.metadata import entry_points  # noqa: PLC0415 - only when enabled

        from hardpoint.core.registry import ENTRY_POINT_GROUP  # noqa: PLC0415

        registry.discover(entry_points(group=ENTRY_POINT_GROUP))

    pricing = PricingTable(config.pricing)
    providers = config.providers
    wrapped: dict[str, Any] = {}
    for kind, path, spec in (
        (Kind.LLM, "providers.llm", providers.llm),
        (Kind.EMBEDDINGS, "providers.embeddings", providers.embeddings),
        (Kind.RERANKER, "providers.reranker", providers.reranker),
    ):
        wrapped[path] = (
            await _create_with_policies(registry, kind, spec, path, config=config, pricing=pricing)
            if spec is not None
            else None
        )

    indexes: dict[str, VectorIndex] = {}
    for name, spec in sorted(config.indexes.items()):
        inner = await _create(
            registry, Kind.INDEX, spec, f"indexes.{name}", {"name": name, "collection": name}
        )
        indexes[name] = PolicyVectorIndex(inner, PolicyChain.from_config(spec.policies))

    state_spec = config.ingestion.state
    state = (
        await _create(registry, Kind.STATE, state_spec, "ingestion.state")
        if state_spec is not None
        else await registry.create(Kind.STATE, "sqlite", {}, config_path="ingestion.state")
    )

    observability = config.observability
    tracer: Tracer = (
        await _create(registry, Kind.TRACER, observability.tracer, "observability.tracer")
        if observability.tracer is not None
        else NoOpTracer()
    )
    if observability.redact:
        tracer = RedactingTracer(tracer, observability.redact)
    metrics: MetricSink = (
        await _create(registry, Kind.METRICS, observability.metrics, "observability.metrics")
        if observability.metrics is not None
        else NoOpMetricSink()
    )
    shared_cache = (
        await _create(registry, Kind.CACHE, config.cache.backend, "cache.backend")
        if config.cache.backend is not None
        else None
    )
    input_checks = [
        await _create_guard(registry, spec, f"guards.input.{position}")
        for position, spec in enumerate(config.guards.input)
    ]
    output_checks = [
        await _create_guard(registry, spec, f"guards.output.{position}")
        for position, spec in enumerate(config.guards.output)
    ]

    return Resources(
        config=config,
        snapshot=resolved.snapshot,
        registry=registry,
        llm_=wrapped["providers.llm"],
        embedder_=wrapped["providers.embeddings"],
        reranker_=wrapped["providers.reranker"],
        indexes=indexes,
        state=state,
        prompts=prompts or _prompt_store(config.project.prompts_dir),
        tracer=tracer,
        metrics=metrics,
        shared_cache=shared_cache,
        pricing=pricing,
        input_checks=tuple(input_checks),
        output_checks=tuple(output_checks),
    )


async def _create_guard(registry: ComponentRegistry, spec: GuardSpec, path: str) -> Any:
    """Construct one guard: its own options plus the ``action`` every guard takes."""
    options = {**spec.options(), "action": spec.action}
    registry.resolve(Kind.GUARD, spec.type, config_path=f"{path}.type")
    return await registry.create(Kind.GUARD, spec.type, options, config_path=path)


def _prompt_store(directory: str) -> PromptStore:
    """Load the project's prompts, or an empty store when it has none."""
    from hardpoint.generation.prompts import (  # noqa: PLC0415 - generation sits above runtime
        FilePromptStore,
        InMemoryPromptStore,
    )

    if Path(directory).is_dir():
        return FilePromptStore(directory)
    return InMemoryPromptStore()


async def build_source(res: Resources, name: str) -> Any:
    """Construct a configured ingestion source.

    Separate from :func:`build_resources` because only ingestion needs a
    source, and constructing one checks its location exists -- which ``ask``
    has no reason to care about.

    Raises:
        ConfigError: If no source of that name is configured.
    """
    spec = res.config.sources.get(name)
    if spec is None:
        raise ConfigError(
            f"No source named {name!r} is configured.",
            config_path=f"sources.{name}",
            remedy=(
                f"Configured sources: {', '.join(sorted(res.config.sources)) or '(none)'}. "
                f"Add `sources.{name}` with `type: local_files` and a `root`."
            ),
        )
    return await _create(res.registry, Kind.SOURCE, spec, f"sources.{name}", {"source_id": name})


# --------------------------------------------------------------------------- #
# The one execution path                                                      #
# --------------------------------------------------------------------------- #


def load_pipeline(res: Resources) -> Pipeline[Any, Any]:
    """Build the project's pipeline through its configured factory.

    ``project.pipeline`` names a ``module:function`` taking :class:`Resources`.
    The working directory is put on ``sys.path`` so a generated project's
    ``pipelines`` package imports without being installed.

    Raises:
        ConfigError: If the factory cannot be imported or found.
    """
    target = res.config.project.pipeline
    module_name, _, function_name = target.partition(":")
    if not module_name or not function_name:
        raise ConfigError(
            f"`project.pipeline` must be `module:function`, got {target!r}.",
            config_path="project.pipeline",
            remedy="For example `project.pipeline: pipelines.rag:build`.",
        )

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    _forget_if_stale(module_name, Path(cwd))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ConfigError(
            f"The pipeline module {module_name!r} could not be imported: {exc}.",
            config_path="project.pipeline",
            remedy=(
                "Run from the project directory, or point `project.pipeline` at an "
                "importable factory such as `hardpoint.recipes.naive:build`."
            ),
            cause=exc,
        ) from exc

    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise ConfigError(
            f"{module_name!r} has no callable {function_name!r}.",
            config_path="project.pipeline",
            remedy=f"Define `def {function_name}(res: Resources) -> Pipeline` in {module_name}.",
        )
    pipeline: Pipeline[Any, Any] = factory(res)
    return pipeline


def _forget_if_stale(module_name: str, project: Path) -> None:
    """Drop a cached project module that belongs to a different project directory.

    ``pipelines.rag`` is a generic name. A process that loads two projects -- a
    test suite, a notebook -- would otherwise keep answering with the first
    project's pipeline. Only a module the current directory itself provides is
    considered, so library recipes are never touched.
    """
    top = module_name.split(".", 1)[0]
    local = project / top
    if not (local.is_dir() or local.with_suffix(".py").is_file()):
        return
    cached = sys.modules.get(top)
    location = getattr(cached, "__file__", None) if cached is not None else None
    if location is None or Path(location).resolve().is_relative_to(project.resolve()):
        return
    for name in [name for name in sys.modules if name == top or name.startswith(f"{top}.")]:
        del sys.modules[name]


async def answer_query(
    pipeline: Pipeline[Any, Any],
    query: Any,
    res: Resources,
    *,
    run_id: str | None = None,
    on_delta: DeltaSink | None = None,
) -> Answer:
    """Run a pipeline and complete its ``Answer`` with what only the run knows.

    The final step returns an ``Answer``; this attaches the run-wide usage (every
    step, not just generation), every degradation, the trace id, and the
    manifest facts -- config hash, index epochs, model ids -- that make the
    answer reproducible (ARCHITECTURE.md §10).

    ``on_delta`` streams the final step's text as it is produced, through the
    same pipeline run: what the service's SSE endpoint uses.

    Raises:
        ContractError: If the pipeline's last step did not return an ``Answer``.
    """
    ctx = res.run_context(run_id=run_id)
    try:
        run = await pipeline.run_detailed(query, ctx, on_delta=on_delta)
    except Exception:
        ctx.metrics.increment("hardpoint.requests", pipeline=pipeline.name, outcome="error")
        raise
    answer = run.value
    if not isinstance(answer, Answer):
        raise ContractError(
            f"Pipeline {pipeline.name!r} returned {type(answer).__name__}, not an Answer.",
            component=pipeline.name,
            remedy="End the pipeline with a step that returns an Answer, such as Generate.",
        )

    outcome = "blocked" if answer.blocked else "abstained" if answer.abstained else "ok"
    ctx.metrics.increment("hardpoint.requests", pipeline=pipeline.name, outcome=outcome)

    model_ids = dict(answer.manifest.model_ids)
    if res.embedder_ is not None:
        model_ids.setdefault("embeddings", res.embedder_.id)

    return answer.model_copy(
        update={
            "usage": ctx.usage.snapshot(),
            "degradations": [*run.degradations, *answer.degradations],
            "trace_id": run.trace_id,
            "run_id": ctx.run_id,
            "manifest": answer.manifest.model_copy(
                update={
                    "config_hash": res.snapshot.hash,
                    "pipeline_name": pipeline.name,
                    "model_ids": model_ids,
                    "index_epochs": await res.epochs(),
                }
            ),
        }
    )
