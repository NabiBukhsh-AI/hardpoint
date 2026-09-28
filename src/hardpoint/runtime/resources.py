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
from hardpoint.observability.tracing import NoOpTracer
from hardpoint.runtime.policies import PolicyChain
from hardpoint.runtime.wrapped import (
    PolicyEmbeddingModel,
    PolicyLanguageModel,
    PolicyReranker,
    PolicyVectorIndex,
)

if TYPE_CHECKING:
    from hardpoint.core.config.loader import ResolvedConfig
    from hardpoint.core.config.schema import ComponentSpec, HardpointConfig
    from hardpoint.core.ports import (
        EmbeddingModel,
        LanguageModel,
        MetricSink,
        PromptStore,
        Reranker,
        StateStore,
        Tracer,
        VectorIndex,
    )
    from hardpoint.runtime.pipeline import Pipeline

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
        tracer: Where spans go.
        metrics: Where measurements go.
        cache: Request-scoped and shared caches.
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
    cache: CacheHandle = field(default_factory=CacheHandle)

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

    def with_components(self, **changes: Any) -> Resources:
        """Return a copy with some components replaced.

        How the CLI wraps the model to capture the rendered prompt for
        ``--explain``, and how the eval runner puts cassettes in front of
        providers -- by substitution, visibly, rather than through a hook.
        """
        return replace(self, **changes)

    def run_context(self, *, run_id: str | None = None) -> RunContext:
        """Build a ``RunContext`` with the configured budget and deadline."""
        budget_config = self.config.budgets.request
        return RunContext(
            run_id=run_id or new_run_id(),
            tracer=self.tracer,
            metrics=self.metrics,
            deadline=Deadline.in_seconds(budget_config.deadline_s),
            budget=Budget.from_config(budget_config),
            cache=self.cache,
            config=self.snapshot,
            usage=UsageAccumulator(),
            extras={},
        )

    async def epochs(self) -> dict[str, int]:
        """Return each configured index's epoch from the manifest.

        The manifest owns the epoch (ADR-009), so it is read there rather than
        from the index.
        """
        await self.state.initialise()
        return {name: await self.state.index_epoch(name) for name in sorted(self.indexes)}

    async def aclose(self) -> None:
        """Close every component that holds a connection or a file."""
        seen: set[int] = set()
        for component in (self.llm_, self.embedder_, self.reranker_, *self.indexes.values()):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            closer = getattr(component, "aclose", None)
            if closer is not None:
                await closer()
        close_state = getattr(self.state, "close", None)
        if close_state is not None:
            await close_state()


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
    registry: ComponentRegistry, kind: Kind, spec: ComponentSpec, path: str
) -> Any:
    """Construct a component and its fallback, and wrap both in the policy chain."""
    inner = await _create(registry, kind, spec, path)
    chain = PolicyChain.from_config(spec.policies)
    fallback_spec = spec.policies.fallback
    fallback = (
        await _create(registry, kind, fallback_spec, f"{path}.policies.fallback")
        if fallback_spec is not None
        else None
    )

    if kind is Kind.LLM:
        return PolicyLanguageModel(inner, chain, fallback)
    if kind is Kind.EMBEDDINGS:
        return PolicyEmbeddingModel(inner, chain, fallback)
    if kind is Kind.RERANKER:
        return PolicyReranker(inner, chain, fallback)
    return PolicyVectorIndex(inner, chain)


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

    providers = config.providers
    llm = (
        await _create_with_policies(registry, Kind.LLM, providers.llm, "providers.llm")
        if providers.llm
        else None
    )
    embedder = (
        await _create_with_policies(
            registry, Kind.EMBEDDINGS, providers.embeddings, "providers.embeddings"
        )
        if providers.embeddings
        else None
    )
    reranker = (
        await _create_with_policies(
            registry, Kind.RERANKER, providers.reranker, "providers.reranker"
        )
        if providers.reranker
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

    return Resources(
        config=config,
        snapshot=resolved.snapshot,
        registry=registry,
        llm_=llm,
        embedder_=embedder,
        reranker_=reranker,
        indexes=indexes,
        state=state,
        prompts=prompts or _prompt_store(config.project.prompts_dir),
    )


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


async def answer_query(
    pipeline: Pipeline[Any, Any], query: Any, res: Resources, *, run_id: str | None = None
) -> Answer:
    """Run a pipeline and complete its ``Answer`` with what only the run knows.

    The final step returns an ``Answer``; this attaches the run-wide usage (every
    step, not just generation), every degradation, the trace id, and the
    manifest facts -- config hash, index epochs, model ids -- that make the
    answer reproducible (ARCHITECTURE.md §10).

    Raises:
        ContractError: If the pipeline's last step did not return an ``Answer``.
    """
    ctx = res.run_context(run_id=run_id)
    run = await pipeline.run_detailed(query, ctx)
    answer = run.value
    if not isinstance(answer, Answer):
        raise ContractError(
            f"Pipeline {pipeline.name!r} returned {type(answer).__name__}, not an Answer.",
            component=pipeline.name,
            remedy="End the pipeline with a step that returns an Answer, such as Generate.",
        )

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
