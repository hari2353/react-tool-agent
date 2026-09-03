"""Unit tests for the TF-IDF retrieval store."""

from __future__ import annotations

import pytest

from tool_agent.retrieval import TfIdfStore, default_store


class TestTfIdfStore:
    def test_exact_term_ranks_first(self) -> None:
        store = default_store()
        hits = store.search("zero tolerance child labor", top_k=2)
        assert hits[0]["doc_id"] == "zero-tolerance"
        assert "zero-tolerance" in str(hits[0]["title"]).lower().replace(" ", "-")

    def test_cadence_query_finds_cadence_doc(self) -> None:
        store = default_store()
        hits = store.search("how often are high risk suppliers audited", top_k=1)
        assert hits[0]["doc_id"] == "audit-cadence"

    def test_top_k_bounds(self) -> None:
        store = default_store()
        assert len(store.search("supplier risk", top_k=2)) == 2
        with pytest.raises(ValueError, match="top_k"):
            store.search("supplier risk", top_k=0)

    def test_scores_sorted_desc(self) -> None:
        store = default_store()
        hits = store.search("supplier corrective action deadlines", top_k=5)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_stopwords_ignored(self) -> None:
        store = default_store()
        a = store.search("the audit cadence", top_k=1)
        b = store.search("audit cadence", top_k=1)
        assert a[0]["doc_id"] == b[0]["doc_id"]

    def test_empty_store_returns_empty(self) -> None:
        assert TfIdfStore().search("anything") == []

    def test_duplicate_doc_id_rejected(self) -> None:
        store = TfIdfStore()
        store.add("d1", "t", "text one")
        with pytest.raises(ValueError, match="duplicate"):
            store.add("d1", "t2", "text two")

    def test_blank_text_rejected(self) -> None:
        store = TfIdfStore()
        with pytest.raises(ValueError, match="required"):
            store.add("d1", "t", "   ")

    def test_snippet_truncated(self) -> None:
        store = TfIdfStore()
        store.add("long", "long doc", "word " * 200)
        (hit,) = store.search("word", top_k=1)
        assert str(hit["snippet"]).endswith("...")
