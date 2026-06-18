from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .paths import AGENT_ARTIFACTS_DIR, POLICIES_DIR, PROJECT_ROOT


EVIDENCE_SOURCES = [
    POLICIES_DIR / "risk_policy.md",
    PROJECT_ROOT / "docs" / "leakage_tradeoff_and_justification.md",
    PROJECT_ROOT / "docs" / "modeling_summary.md",
]


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_path: str
    source_name: str
    section: str
    text: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_markdown_sections(path: Path) -> list[EvidenceChunk]:
    text = path.read_text(encoding="utf-8")
    current_section = path.stem
    current_lines: list[str] = []
    chunks: list[EvidenceChunk] = []

    def flush() -> None:
        if not current_lines:
            return
        body = _clean("\n".join(current_lines))
        if not body:
            return
        chunk_id = f"{path.stem}:{len(chunks) + 1:02d}"
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                source_path=str(path.relative_to(PROJECT_ROOT)),
                source_name=path.name,
                section=current_section,
                text=body,
            )
        )

    for line in text.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush()
            current_section = heading.group(2).strip()
            current_lines = []
            continue
        if line.strip():
            current_lines.append(line)
    flush()
    return chunks


def build_evidence_chunks() -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for source in EVIDENCE_SOURCES:
        chunks.extend(parse_markdown_sections(source))
    return chunks


def write_evidence_store(output_dir: Path = AGENT_ARTIFACTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_evidence_chunks()
    path = output_dir / "evidence_store.json"
    path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_evidence_store(path: Path | None = None) -> list[EvidenceChunk]:
    store_path = path or (AGENT_ARTIFACTS_DIR / "evidence_store.json")
    if not store_path.exists():
        write_evidence_store(store_path.parent)
    data = json.loads(store_path.read_text(encoding="utf-8"))
    return [EvidenceChunk(**item) for item in data]


def retrieve_evidence(query: str, top_k: int = 2) -> list[dict[str, object]]:
    chunks = load_evidence_store()
    corpus = [f"{chunk.section}. {chunk.text}" for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "source_name": chunk.source_name,
                "section": chunk.section,
                "text": chunk.text,
                "score": float(score),
            }
        )
    return results

