"""Prompt storage, rendering and versioning.

Implements the ``PromptStore`` side of INSTRUCTIONS.md §6.4.

## The library ships no prompt content

**[LOCKED]** INSTRUCTIONS.md §13.11. Prompts are the product's voice: they belong
in the repository, versioned, diffable and evaluable. A prompt shipped in a
library goes stale, cannot be tested against anyone's domain, and turns the one
thing a team most needs to iterate on into something they have to fork a
dependency to change.

What ships here is the *mechanism*: somewhere to keep prompts, a renderer, and a
version derived from the content.

## The version is a content hash, and that is not an implementation detail

Every generation cache key includes the prompt version (ARCHITECTURE.md §22.2).
If the version did not move when the text moved, editing a prompt would keep
serving answers generated from the previous one, indefinitely, with nothing to
suggest anything was wrong. Deriving it from the content makes that impossible
to get wrong -- there is no separate number anyone can forget to bump.

## The renderer is restricted on purpose

``{{ name }}`` substitution and nothing else. No expressions, no attribute
access, no loops, no filters.

``str.format`` is not usable here: ``"{x.__class__.__init__.__globals__}"``
reaches the interpreter from a template, and templates are exactly the kind of
file that gets edited by people who are not thinking about sandboxing. A
restricted renderer that refuses is better than a powerful one that has to be
audited.

A user who genuinely needs Jinja passes a callable; an engine abstraction here
would earn nothing (ARCHITECTURE.md §9.3).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from hardpoint.core.errors import ConfigError
from hardpoint.core.ids import content_hash
from hardpoint.core.ports import Message, RenderedPrompt
from hardpoint.core.types import JsonValue

__all__ = [
    "PLACEHOLDER",
    "FilePromptStore",
    "InMemoryPromptStore",
    "PromptTemplate",
    "render_template",
]

PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
"""``{{ name }}``. A bare identifier and nothing else -- see the module docstring."""

_ROLE_HEADER = re.compile(r"^---\s*(system|user|assistant)\s*---\s*$", re.IGNORECASE)


def render_template(template: str, variables: Mapping[str, JsonValue]) -> str:
    """Substitute ``{{ name }}`` placeholders.

    Args:
        template: The template text.
        variables: What to substitute.

    Returns:
        The rendered text.

    Raises:
        ConfigError: If the template references a variable that was not
            supplied. A missing variable renders as an empty hole in a prompt,
            which produces a confidently wrong answer rather than an error, so
            it fails here instead.
    """
    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            missing.append(name)
            return ""
        value = variables[name]
        return value if isinstance(value, str) else str(value)

    rendered = PLACEHOLDER.sub(substitute, template)

    if missing:
        raise ConfigError(
            f"The prompt references {', '.join(sorted(set(missing)))}, which was not supplied.",
            remedy=(
                f"Pass {sorted(set(missing))[0]!r} when rendering the prompt, or "
                f"remove the placeholder. An unsubstituted variable would leave a "
                f"hole in the prompt and produce a confidently wrong answer."
            ),
        )
    return rendered


class PromptTemplate:
    """One prompt: a sequence of role-tagged sections, and a version.

    A template is plain text, optionally divided into roles with ``--- system ---``
    and ``--- user ---`` markers. Text before any marker is the user message,
    because that is the safe default: retrieved content must never reach a system
    message (INSTRUCTIONS.md §6.4 **[LOCKED]**), and a template with no markers is
    most often a body with a question in it.

    Args:
        name: The prompt's name.
        text: The template.

    Attributes:
        version: First twelve hex characters of the content hash. Stable across
            machines, and it moves whenever the text does.
    """

    __slots__ = ("_sections", "name", "text", "version")

    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text
        self.version = content_hash(text.encode("utf-8"))[:12]
        self._sections = _split_roles(text)

    def render(self, variables: Mapping[str, JsonValue]) -> RenderedPrompt:
        """Render every section, dropping any that came out empty.

        Args:
            variables: What to substitute.

        Returns:
            The rendered messages plus this template's version.

        Raises:
            ConfigError: On an unsupplied variable.
        """
        messages = tuple(
            Message(role=role, content=body)  # type: ignore[arg-type]  # role is validated by the splitter
            for role, body in (
                (role, render_template(section, variables).strip())
                for role, section in self._sections
            )
            if body
        )
        return RenderedPrompt(name=self.name, version=self.version, messages=messages)

    def variables(self) -> frozenset[str]:
        """Return every placeholder the template references.

        What ``doctor`` checks a prompt's inputs against, so a template asking
        for a variable nothing supplies is caught at startup.
        """
        return frozenset(match.group(1) for match in PLACEHOLDER.finditer(self.text))

    def __repr__(self) -> str:
        """Render the name and version."""
        return f"PromptTemplate(name={self.name!r}, version={self.version!r})"


def _split_roles(text: str) -> list[tuple[str, str]]:
    """Split a template into ``(role, body)`` sections.

    Text before the first marker becomes a user message. See
    :class:`PromptTemplate` for why that default and not ``system``.
    """
    sections: list[tuple[str, str]] = []
    role = "user"
    body: list[str] = []

    for line in text.splitlines():
        header = _ROLE_HEADER.match(line.strip())
        if header:
            if body:
                sections.append((role, "\n".join(body)))
                body = []
            role = header.group(1).lower()
            continue
        body.append(line)

    if body:
        sections.append((role, "\n".join(body)))
    return sections or [("user", text)]


class InMemoryPromptStore:
    """Prompts held in a mapping.

    For tests, and for a project that would rather keep prompts in Python than
    in files. Not a fake: it is a complete ``PromptStore``, and it is what the
    generated project's tests bind their pipeline to.

    Args:
        prompts: Name to template text.
    """

    def __init__(self, prompts: Mapping[str, str] | None = None) -> None:
        self._templates = {
            name: PromptTemplate(name, text) for name, text in (prompts or {}).items()
        }

    def add(self, name: str, text: str) -> PromptTemplate:
        """Add or replace a prompt, returning the compiled template."""
        template = PromptTemplate(name, text)
        self._templates[name] = template
        return template

    async def render(
        self, name: str, variables: Mapping[str, JsonValue], *, version: str | None = None
    ) -> RenderedPrompt:
        """Render a named prompt.

        Raises:
            ConfigError: If the prompt is unknown, or if ``version`` names a
                version this store does not hold. Silently rendering a different
                version than the caller asked for would make a reproduction
                attempt quietly meaningless.
        """
        template = self._require(name)
        if version is not None and version != template.version:
            raise ConfigError(
                f"Prompt {name!r} is at version {template.version!r}, but "
                f"{version!r} was requested.",
                remedy=(
                    "Check out the revision of the prompt that produced the run "
                    "you are reproducing. Prompt versions are content hashes, so "
                    "a mismatch means the text differs."
                ),
            )
        return template.render(variables)

    async def version_of(self, name: str) -> str:
        """Return a prompt's current version, without rendering it.

        Called to build a cache key on every request, so it must stay cheap.
        """
        return self._require(name).version

    def names(self) -> list[str]:
        """Return every prompt name, sorted."""
        return sorted(self._templates)

    def _require(self, name: str) -> PromptTemplate:
        template = self._templates.get(name)
        if template is None:
            raise ConfigError(
                f"No prompt named {name!r}.",
                component=name,
                remedy=(
                    f"Available prompts: {', '.join(self.names()) or '(none)'}. "
                    f"Prompts live in the generated project, not in the library."
                ),
            )
        return template

    def __repr__(self) -> str:
        """Render how many prompts are held."""
        return f"InMemoryPromptStore(prompts={len(self._templates)})"


class FilePromptStore(InMemoryPromptStore):
    """Prompts loaded from a directory of text files.

    What a generated project uses: ``prompts/answer.md`` becomes the prompt
    ``answer``. Files, so prompts are diffable in review and versioned by the
    same history as the code that uses them.

    Loaded once at construction rather than per render. Re-reading on every
    request would make a prompt's version change under a running process, and
    the run manifest would then record a version that no longer describes what
    was actually sent.

    Args:
        directory: Where the prompt files are.
        suffixes: Extensions to load.

    Raises:
        ConfigError: If the directory does not exist. A prompt store pointed at
            nothing would fail on the first request instead of at startup.
    """

    def __init__(
        self, directory: str | Path, *, suffixes: tuple[str, ...] = (".md", ".txt", ".prompt")
    ) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ConfigError(
                f"The prompt directory {self.directory} does not exist.",
                config_path="prompts",
                remedy=(
                    f"Create {self.directory} and add your prompt files, or point the "
                    f"prompt store at the directory holding them. The library ships "
                    f"no prompt content: prompts are a product asset."
                ),
            )

        super().__init__(
            {
                path.stem: path.read_text(encoding="utf-8")
                for path in sorted(self.directory.iterdir())
                if path.is_file() and path.suffix.lower() in suffixes
            }
        )

    def __repr__(self) -> str:
        """Render the directory and how many prompts were loaded."""
        return f"FilePromptStore(directory={str(self.directory)!r}, prompts={len(self.names())})"
