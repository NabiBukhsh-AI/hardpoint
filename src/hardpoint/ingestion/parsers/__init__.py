"""Document parsers mapping source blobs to ParsedDocument.

M1 ships the two formats every corpus has: plain text and Markdown. The
``parsers`` extra covers the formats that genuinely need a library (PDF, DOCX,
HTML), and those adapters are M6.
"""

from __future__ import annotations

from hardpoint.ingestion.parsers.text import MarkdownParser, TextParser, decode

__all__ = ["MarkdownParser", "TextParser", "decode"]
