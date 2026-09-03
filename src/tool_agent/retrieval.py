"""Minimal TF-IDF retrieval store - zero dependencies, deterministic.

The knowledge base ships a small synthetic supplier-compliance corpus
(same domain as my ESG pipeline repo) so `kb_search` returns real passages
in the demo. Sublinear TF + L2 normalization + cosine similarity: the
classic, explainable ranking baseline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "this", "to", "was", "were", "will", "with",
    )
)


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS
    ]


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrievable knowledge-base passage."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked search result."""

    doc_id: str
    title: str
    snippet: str
    score: float


@dataclass
class TfIdfStore:
    """In-memory TF-IDF store with sublinear TF and cosine ranking."""

    passages: list[Passage] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _vectors: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def add(self, doc_id: str, title: str, text: str) -> None:
        if not doc_id or not text.strip():
            msg = "doc_id and text are required"
            raise ValueError(msg)
        if any(p.doc_id == doc_id for p in self.passages):
            msg = f"duplicate doc_id: {doc_id}"
            raise ValueError(msg)
        self.passages.append(Passage(doc_id=doc_id, title=title, text=text))
        self._fitted = False

    def fit(self) -> None:
        """Compute document frequencies and L2-normalized TF-IDF vectors."""
        self._df = {}
        for passage in self.passages:
            for term in set(_tokenize(passage.text)):
                self._df[term] = self._df.get(term, 0) + 1
        self._vectors = [self._vectorize(p.text) for p in self.passages]
        self._fitted = True

    def _vectorize(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        n_docs = max(len(self.passages), 1)
        vector: dict[str, float] = {}
        for term, count in counts.items():
            tf = 1.0 + math.log(count)
            idf = math.log((1 + n_docs) / (1 + self._df.get(term, 0))) + 1.0
            vector[term] = tf * idf
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {k: v / norm for k, v in vector.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        """Cosine-ranked hits; returns a JSON-friendly dict per hit."""
        if not self._fitted:
            self.fit()
        if not self.passages:
            return []
        if top_k < 1:
            msg = "top_k must be >= 1"
            raise ValueError(msg)
        q = self._vectorize(query)
        scored: list[tuple[float, int]] = []
        for idx, vec in enumerate(self._vectors):
            score = sum(weight * vec.get(term, 0.0) for term, weight in q.items())
            scored.append((score, idx))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        hits: list[dict[str, object]] = []
        for score, idx in scored[:top_k]:
            passage = self.passages[idx]
            snippet = (
                passage.text[:280] + ("..." if len(passage.text) > 280 else "")
            )
            hits.append(
                {
                    "doc_id": passage.doc_id,
                    "title": passage.title,
                    "snippet": snippet,
                    "score": round(score, 4),
                }
            )
        return hits


KB_CORPUS: tuple[tuple[str, str, str], ...] = (
    (
        "audit-cadence",
        "Supplier audit cadence policy",
        "High-risk suppliers are audited every 12 months. Medium-risk suppliers "
        "are audited every 24 months. Low-risk suppliers complete a desktop "
        "self-assessment every 36 months. Any supplier with an open critical "
        "corrective action is re-audited within 6 months regardless of band.",
    ),
    (
        "zero-tolerance",
        "Zero-tolerance findings policy",
        "Findings in child labor or forced labor are zero-tolerance. Any "
        "zero-tolerance finding immediately suspends new purchase orders, "
        "freezes the supplier scorecard, and requires escalation to the ethics "
        "committee within 5 business days.",
    ),
    (
        "capa-deadlines",
        "Corrective action deadlines",
        "Corrective actions follow severity-based deadlines: critical findings "
        "are due within 7 calendar days, major findings within 30 days, and "
        "minor findings within 60 days. Overdue critical actions are escalated "
        "to the sourcing director weekly.",
    ),
    (
    "risk-bands",
        "Supplier risk bands",
        "Suppliers are banded low, medium, high, or critical based on the "
        "weighted risk score. Scores above 60 land in the high band and above "
        "85 in the critical band. Band drives audit cadence, sourcing "
        "approval limits, and payment terms.",
    ),
    (
    "onboarding",
        "New supplier onboarding requirements",
        "New suppliers must pass a documentation screen, sign the supplier "
        "code of conduct, and complete a screening questionnaire before the "
        "first purchase order. High-spend categories additionally require a "
        "pre-award on-site audit.",
    ),
)


def default_store() -> TfIdfStore:
    """Knowledge base preloaded with the synthetic compliance corpus."""
    store = TfIdfStore()
    for doc_id, title, text in KB_CORPUS:
        store.add(doc_id, title, text)
    store.fit()
    return store
