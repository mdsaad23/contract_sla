"""
Retrieval — runs the SLA-targeted queries against a per-document index.

Uses the shared embedder singleton from pipeline.embedder (no double-loading).
"""

from typing import List
from llama_index.core import VectorStoreIndex
from pipeline.embedder import get_embed_model

SLA_QUERIES = [
    # Performance SLAs
    "uptime guarantee availability SLA percentage service level",
    "response time incident support SLA hours priority",
    "SLA breach threshold availability falls below measurement period",

    # Monetary penalties
    "penalty monetary damages liquidated cash fine",
    "service credit uptime breach penalty formula",
    "late delivery penalty liquidated damages milestones",
    "early termination fee cancellation penalty",
    "late payment interest overdue invoice fee",
    "data breach fine liability security incident penalty",

    # Contract mechanics
    "renewal auto-renewal notice period term extension",
    "termination notice period cancellation conditions",
    "limitation of liability cap maximum damages aggregate",
    "governing law jurisdiction applicable law",
    "dispute resolution arbitration mediation",

    # Phrase-level anchors — boilerplate phrase matching
    "this agreement shall be governed by the laws",
    "in the event of any dispute between the parties",
    "neither party shall be liable for indirect consequential",
    "this agreement shall automatically renew",
    "any dispute arising out of or in connection with",
]


def retrieve_sla_chunks(index: VectorStoreIndex, top_k: int = 5) -> str:
    """Run all SLA queries, dedupe results, return joined context."""
    embed_model = get_embed_model()
    retriever   = index.as_retriever(similarity_top_k=top_k, embed_model=embed_model)

    seen: set       = set()
    all_chunks: List[str] = []

    for query in SLA_QUERIES:
        nodes = retriever.retrieve(query)
        for node in nodes:
            content = node.get_content()
            if content not in seen:
                seen.add(content)
                all_chunks.append(content)

    return "\n\n---\n\n".join(all_chunks)
