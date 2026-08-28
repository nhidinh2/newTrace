"""Cosine ordering, TF-IDF, reranking, explanations and ranking metrics."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newstrace.evaluation.retrieval import (
    bootstrap_intervals,
    bootstrap_mean_ci,
    chronological_split,
    ndcg_at_k,
    paired_bootstrap_delta_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    topk_overlap,
)
from newstrace.models import Article
from newstrace.retrieval.exact import (
    DenseRetriever,
    ScoredHit,
    TfidfRetriever,
    cosine_scores,
    recency_weight,
    top_k,
)
from newstrace.retrieval.explain import evidence_excerpt, explain_hit
from newstrace.retrieval.rerank import blend_pagerank, rerank_with_signals
from newstrace.utils import utcnow


def test_cosine_scores_order_by_similarity() -> None:
    matrix = np.array([[1, 0], [0.7071, 0.7071], [0, 1]], dtype=np.float32)
    scores = cosine_scores(np.array([1, 0], dtype=np.float32), matrix)
    assert scores[0] > scores[1] > scores[2]
    assert np.isclose(scores[0], 1.0, atol=1e-6)


def test_cosine_scores_on_empty_matrix() -> None:
    assert cosine_scores(np.ones(3, dtype=np.float32), np.zeros((0, 3), dtype=np.float32)).size == 0


def test_top_k_is_deterministic_under_ties() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.1], dtype=np.float32)
    assert list(top_k(scores, 3)) == [0, 1, 2]


def test_top_k_clamps_to_available_items() -> None:
    assert len(top_k(np.array([0.2, 0.1], dtype=np.float32), 10)) == 2


def test_dense_retriever_orders_and_filters(populated_session: Session) -> None:
    from newstrace.representations.registry import load_vectors

    vectors = load_vectors(populated_session, method="full")
    retriever = DenseRetriever(vectors.article_ids, vectors.matrix)
    hits = retriever.search("Atlas 3 agent model benchmark", k=3)
    assert hits
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert all("cosine" in h.signals for h in hits)

    allowed = {hits[-1].article_id}
    filtered = retriever.search("Atlas 3 agent model benchmark", k=3, allowed_ids=allowed)
    assert [h.article_id for h in filtered] == list(allowed)


def test_tfidf_retriever_finds_lexical_matches(populated_session: Session) -> None:
    articles = populated_session.query(Article).all()
    retriever = TfidfRetriever([a.id for a in articles], [a.body_text for a in articles])
    hits = retriever.search("gigawatts data centre demand", k=3)
    assert hits
    best = populated_session.get(Article, hits[0].article_id)
    assert best is not None and "gigawatt" in best.body_text.lower()


def test_tfidf_returns_nothing_for_unmatched_query(populated_session: Session) -> None:
    articles = populated_session.query(Article).all()
    retriever = TfidfRetriever([a.id for a in articles], [a.body_text for a in articles])
    assert retriever.search("zzzz qqqq unmatched vocabulary", k=5) == []


def test_recency_weight_decays() -> None:
    now = utcnow()
    assert recency_weight(now, now, 48) == 1.0
    assert recency_weight(now - timedelta(hours=48), now, 48) == pytest.approx(0.5)
    assert recency_weight(None, now, 48) == 1.0


def test_signal_rerank_penalises_duplicates(populated_session: Session) -> None:
    articles = {a.id: a for a in populated_session.query(Article).all()}
    duplicate = next(a for a in articles.values() if a.is_near_duplicate)
    original = next(a for a in articles.values() if not a.is_near_duplicate)
    hits = [
        ScoredHit(duplicate.id, 0.9, 1, "cosine:full", {"cosine": 0.9}),
        ScoredHit(original.id, 0.9, 2, "cosine:full", {"cosine": 0.9}),
    ]
    reranked = rerank_with_signals(hits, articles, recency_weight_factor=0.0)
    assert reranked[0].article_id == original.id
    assert reranked[-1].signals["duplicate_penalty"] < 1.0


def test_blend_pagerank_changes_order() -> None:
    hits = [
        ScoredHit(1, 0.80, 1, "cosine:full", {"cosine": 0.80}),
        ScoredHit(2, 0.78, 2, "cosine:full", {"cosine": 0.78}),
    ]
    blended = blend_pagerank(hits, {1: 0.001, 2: 0.900}, alpha=0.5)
    assert blended[0].article_id == 2
    assert "pagerank" in blended[0].signals
    assert blended[0].method.endswith("+ppr")


def test_blend_pagerank_on_empty_hits() -> None:
    assert blend_pagerank([], {}) == []


def test_explanation_names_signals_and_flags_duplicates(populated_session: Session) -> None:
    duplicate = populated_session.query(Article).filter(Article.is_near_duplicate.is_(True)).one()
    hit = ScoredHit(duplicate.id, 0.8, 1, "cosine:full", {"cosine": 0.8, "recency": 0.5})
    text = explain_hit(hit, duplicate, "Atlas 3 benchmark")
    assert "semantic similarity" in text
    assert "recency" in text
    assert "near-duplicate" in text
    assert "rank 1" in text


def test_evidence_excerpt_prefers_query_terms(populated_session: Session) -> None:
    article = (
        populated_session.query(Article).filter(Article.source_domain == "gamma.example").one()
    )
    excerpt = evidence_excerpt(article, "gigawatts demand")
    assert "gigawatt" in excerpt.lower()


def test_ranking_metrics() -> None:
    ranked = [3, 1, 2, 4]
    relevant = {1, 2}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank([9, 8], relevant) == 0.0
    assert ndcg_at_k([1, 2], relevant, 2) == pytest.approx(1.0)
    assert ndcg_at_k([3, 4], relevant, 2) == 0.0
    assert topk_overlap([1, 2, 3], [1, 2, 9], 3) == pytest.approx(2 / 3)


def test_ranking_metrics_edge_cases() -> None:
    assert precision_at_k([], set(), 0) == 0.0
    assert recall_at_k([1], set(), 5) == 0.0
    assert topk_overlap([], [], 5) == 0.0


def test_chronological_split_is_ordered_and_leak_free(populated_session: Session) -> None:
    articles = populated_session.query(Article).all()
    split = chronological_split(articles)
    split.assert_no_leakage()
    assert len(split.fit_ids) + len(split.validation_ids) + len(split.test_ids) == len(articles)
    assert not set(split.fit_ids) & set(split.test_ids)
    if split.fit_cutoff and split.test_ids:
        latest_fit = split.fit_cutoff
        test_times = [
            a.published_at for a in articles if a.id in set(split.test_ids) and a.published_at
        ]
        assert all(t.replace(tzinfo=latest_fit.tzinfo) >= latest_fit for t in test_times)


def test_chronological_split_on_empty_input() -> None:
    split = chronological_split([])
    assert split.fit_ids == [] and split.test_ids == []
    split.assert_no_leakage()


# -- bootstrap intervals ----------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
    first = bootstrap_mean_ci(values, seed=549)
    second = bootstrap_mean_ci(values, seed=549)
    assert first == second, "a seeded bootstrap must reproduce exactly"
    assert first["lo"] < first["mean"] < first["hi"]
    assert first["n"] == len(values)


def test_bootstrap_ci_of_a_constant_sample_has_zero_width() -> None:
    interval = bootstrap_mean_ci([0.5] * 10)
    assert interval["lo"] == interval["hi"] == interval["mean"] == 0.5


def test_bootstrap_ci_degenerate_inputs() -> None:
    assert bootstrap_mean_ci([])["n"] == 0.0
    single = bootstrap_mean_ci([0.25])
    assert single["lo"] == single["hi"] == 0.25


def test_bootstrap_ci_narrows_as_the_sample_grows() -> None:
    rng = np.random.default_rng(0)
    sample = rng.uniform(size=400).tolist()
    small = bootstrap_mean_ci(sample[:20], seed=1)
    large = bootstrap_mean_ci(sample, seed=1)
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_paired_bootstrap_detects_a_real_difference() -> None:
    baseline = [0.5] * 40
    worse = [0.2] * 40
    delta = paired_bootstrap_delta_ci(worse, baseline)
    assert delta["hi"] < 0, "a uniformly worse method must have a negative interval"
    assert delta["crosses_zero"] == 0.0


def test_paired_bootstrap_reports_indistinguishable_methods() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.uniform(size=60).tolist()
    jittered = [v + (0.01 if i % 2 else -0.01) for i, v in enumerate(baseline)]
    delta = paired_bootstrap_delta_ci(jittered, baseline)
    assert delta["crosses_zero"] == 1.0
    assert delta["lo"] <= 0.0 <= delta["hi"]


def test_paired_bootstrap_rejects_misaligned_inputs() -> None:
    assert paired_bootstrap_delta_ci([0.1, 0.2], [0.1])["n"] == 0.0


def test_bootstrap_intervals_pairs_rows_by_query() -> None:
    rows = [
        {"query": "a", "ndcg_at_10": 0.4, "recall_at_10": 0.5},
        {"query": "b", "ndcg_at_10": 0.6, "recall_at_10": 0.7},
    ]
    baseline = [
        {"query": "a", "ndcg_at_10": 0.9, "recall_at_10": 1.0},
        {"query": "b", "ndcg_at_10": 0.9, "recall_at_10": 1.0},
        {"query": "unmatched", "ndcg_at_10": 0.1, "recall_at_10": 0.1},
    ]
    intervals = bootstrap_intervals(rows, baseline_rows=baseline)
    assert intervals["ndcg_at_10"]["n"] == 2
    assert intervals["ndcg_at_10_delta"]["n"] == 2, "only queries in both sets are paired"
    assert intervals["ndcg_at_10_delta"]["hi"] < 0


def test_bootstrap_intervals_without_a_baseline_has_no_deltas() -> None:
    rows = [{"query": "a", "ndcg_at_10": 0.4}, {"query": "b", "ndcg_at_10": 0.6}]
    intervals = bootstrap_intervals(rows)
    assert "ndcg_at_10" in intervals
    assert not any(key.endswith("_delta") for key in intervals)
