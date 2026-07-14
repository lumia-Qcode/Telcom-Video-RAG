"""
Orchestrates a single query end-to-end:

  1. Parse the query for structured filters (date/time/labels/verdict).
  2. Hybrid-retrieve (semantic + BM25 + RRF) the most relevant videos in full detail.
  3. Always attach a compact structured index of *every* video (so aggregate/
     count/compare questions - which retrieval alone can't guarantee full
     coverage for - are still answered correctly) plus the overall metrics.
  4. Send everything to the LLM and return the answer + the sources used.

The corpus here is small (tens of videos), so sending a compact one-line-
per-video index alongside the top-k detailed excerpts is cheap and makes
aggregate questions ("how many videos have technicians?", "what's overall
precision?") reliable without needing a separate SQL-like query layer.
"""

from dataclasses import dataclass
from typing import List

from data_loader import load_overall_metrics, load_video_records
from llm_client import generate_answer
from query_parsing import parse_query
from retriever import HybridRetriever, RetrievedDoc

_records_cache = None
_metrics_cache = None
_retriever_singleton = None


def _get_retriever() -> HybridRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = HybridRetriever()
    return _retriever_singleton


def _get_records():
    global _records_cache
    if _records_cache is None:
        _records_cache = load_video_records()
    return _records_cache


def _get_metrics():
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = load_overall_metrics()
    return _metrics_cache


def _compact_index() -> str:
    lines = ["FULL VIDEO INDEX (one line per video, covers all videos - use this for descriptive/attribute matching and any counting/listing/comparison):"]
    for r in _get_records():
        lines.append(
            f"- {r.video} | date: {r.date} {r.time} | labels: {', '.join(r.expected_labels) or 'none'} "
            f"| classification: {r.classification} | verdict: {r.verdict or 'n/a'} "
            f"| summary: {r.summary_caption}"
        )
    return "\n".join(lines)


def _metrics_block() -> str:
    metrics = _get_metrics()
    if not metrics:
        return ""
    lines = ["OVERALL PIPELINE METRICS:"]
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _detailed_context(docs: List[RetrievedDoc]) -> str:
    if not docs:
        return "DETAILED EXCERPTS: (no closely matching videos found for this query)"
    parts = ["DETAILED EXCERPTS (most relevant videos to this specific query):"]
    for d in docs:
        parts.append("---")
        parts.append(d.document)
    return "\n".join(parts)


@dataclass
class RagResult:
    answer: str
    sources: List[RetrievedDoc]


def answer_query(query: str) -> RagResult:
    filters = parse_query(query)
    retriever = _get_retriever()
    docs = retriever.retrieve(query, filters=filters)

    context = "\n\n".join([_compact_index(), _metrics_block(), _detailed_context(docs)])
    answer = generate_answer(query, context)

    return RagResult(answer=answer, sources=docs)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many videos show a technician?"
    result = answer_query(q)
    print("Q:", q)
    print("\nA:", result.answer)
    print("\nSources:")
    for d in result.sources:
        print(f"  - {d.video}: {d.metadata.get('summary_caption', '')}")
