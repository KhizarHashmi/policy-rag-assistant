"""Structure-aware chunking.

Policy documents are organised by headings, and one heading usually answers one question
("Maternity", "Deductible and Co-insurance"). So we split on headings first and only split
further (on paragraph boundaries) when a section is too long. Each chunk is prefixed with a
contextual header (document title, plan, section path). Without it, a chunk that says
"covered up to SGD 450 per day" doesn't say WHICH plan, and retrieval can't tell
Essential, Plus and Premier apart.
"""
import re
from dataclasses import dataclass
from pathlib import Path

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    plan: str
    section: str
    body: str

    @property
    def text(self) -> str:
        """What gets embedded and indexed: contextual header + body."""
        return f"{self.title} | plan: {self.plan} | section: {self.section}\n{self.body}"

    def metadata(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "plan": self.plan, "section": self.section}


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, raw[m.end():]


def _split_long(body: str, max_chars: int) -> list[str]:
    """Greedy paragraph packing; never splits inside a paragraph unless it alone exceeds max_chars."""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    parts, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            parts.append(cur)
            cur = ""
        while len(p) > max_chars:  # pathological long paragraph: hard split on sentence boundary
            cut = p.rfind(". ", 0, max_chars) + 1 or max_chars
            parts.append(p[:cut].strip())
            p = p[cut:].strip()
        cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def chunk_markdown(raw: str, doc_id: str, max_chars: int = 900) -> list[Chunk]:
    meta, body = parse_frontmatter(raw)
    title, plan = meta.get("title", doc_id), meta.get("plan", "all")

    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    for line in body.splitlines():
        h = HEADING.match(line)
        if h:
            level = len(h.group(1))
            path = path[: level - 1] + [h.group(2).strip()]
            sections.append((" > ".join(path), []))
        elif sections:
            sections[-1][1].append(line)

    chunks = []
    for section, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        for i, part in enumerate(_split_long(text, max_chars)):
            slug = re.sub(r"[^a-z0-9]+", "-", section.split(" > ")[-1].lower()).strip("-")
            chunks.append(Chunk(f"{doc_id}::{slug}::{i}", doc_id, title, plan, section, part))
    return chunks


def load_corpus(docs_dir: Path, max_chars: int = 900) -> list[Chunk]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.stem, max_chars)
    return chunks
