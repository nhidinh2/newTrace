"""News graph construction and PageRank normalisation/convergence."""

from __future__ import annotations

import networkx as nx
import pytest
from sqlalchemy.orm import Session

from newstrace.graph.build import (
    EDGE_BELONGS_TO,
    EDGE_NEAR_DUPLICATE_OF,
    EDGE_PUBLISHED_BY,
    NODE_ARTICLE,
    NODE_SOURCE,
    article_node,
    build_graph,
    graph_stats,
    parse_node,
    source_node,
)
from newstrace.graph.pagerank import (
    damping_sensitivity,
    personalized_pagerank,
    seed_from_articles,
)
from newstrace.models import Article


def test_node_helpers_roundtrip() -> None:
    assert parse_node(article_node(7)) == (NODE_ARTICLE, "7")
    assert parse_node(source_node("Example.COM")) == (NODE_SOURCE, "example.com")


def test_graph_contains_expected_node_and_edge_types(populated_session: Session) -> None:
    graph = build_graph(populated_session)
    stats = graph_stats(graph)
    assert stats.nodes > 0 and stats.edges > 0
    assert stats.node_type_counts[NODE_ARTICLE] == populated_session.query(Article).count()
    assert NODE_SOURCE in stats.node_type_counts
    kinds = {data["kind"] for _, _, data in graph.edges(data=True)}
    assert EDGE_PUBLISHED_BY in kinds
    assert EDGE_BELONGS_TO in kinds
    assert EDGE_NEAR_DUPLICATE_OF in kinds


def test_duplicate_edges_are_down_weighted(populated_session: Session) -> None:
    graph = build_graph(populated_session, duplicate_edge_weight=0.2)
    weights = [
        data["weight"]
        for _, _, data in graph.edges(data=True)
        if data["kind"] == EDGE_NEAR_DUPLICATE_OF
    ]
    assert weights and all(w == pytest.approx(0.2) for w in weights)


def test_pagerank_is_a_probability_distribution(populated_session: Session) -> None:
    graph = build_graph(populated_session)
    result = personalized_pagerank(graph)
    assert result.total_mass() == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0 for v in result.scores.values())
    assert result.converged
    assert result.iterations >= 1


def test_personalization_concentrates_mass(populated_session: Session) -> None:
    graph = build_graph(populated_session)
    articles = populated_session.query(Article).order_by(Article.id).all()
    seeds = seed_from_articles([articles[0].id])
    seeded = personalized_pagerank(graph, personalization=seeds)
    uniform = personalized_pagerank(graph)
    target = article_node(articles[0].id)
    assert seeded.scores[target] > uniform.scores[target]
    assert seeded.total_mass() == pytest.approx(1.0, abs=1e-9)


def test_dangling_nodes_do_not_leak_mass() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_node("c")  # isolated
    graph.add_node("d")  # dangling sink target
    graph.add_edge("b", "d", weight=1.0)
    result = personalized_pagerank(graph)
    assert result.dangling_nodes >= 1
    assert result.total_mass() == pytest.approx(1.0, abs=1e-9)


def test_disconnected_components_are_handled() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("x", "y", weight=1.0)
    stats = graph_stats(graph)
    assert stats.connected_components == 2
    assert 0 < stats.largest_component_fraction <= 1.0
    result = personalized_pagerank(graph, personalization={"a": 1.0})
    assert result.total_mass() == pytest.approx(1.0, abs=1e-9)
    assert result.scores["a"] > result.scores["x"]


def test_empty_graph_is_safe() -> None:
    result = personalized_pagerank(nx.DiGraph())
    assert result.scores == {}
    assert result.converged
    stats = graph_stats(nx.DiGraph())
    assert stats.nodes == 0 and stats.connected_components == 0


def test_higher_damping_needs_more_iterations() -> None:
    # A ring is already stationary under the uniform vector, so it converges in
    # one step at any damping. Use an asymmetric graph with a seeded restart.
    graph = nx.DiGraph()
    for i in range(12):
        graph.add_edge(f"n{i}", f"n{(i + 1) % 12}", weight=1.0)
    graph.add_edge("n0", "hub", weight=3.0)
    graph.add_edge("hub", "n5", weight=1.0)
    seeds = {"n0": 1.0}
    low = personalized_pagerank(graph, personalization=seeds, damping=0.5)
    high = personalized_pagerank(graph, personalization=seeds, damping=0.95)
    assert high.iterations > low.iterations
    assert low.converged and high.converged


def test_damping_sensitivity_reports_each_setting() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "a", weight=1.0)
    rows = damping_sensitivity(graph, dampings=(0.5, 0.85))
    assert [r["damping"] for r in rows] == [0.5, 0.85]
    assert all(r["total_mass"] == pytest.approx(1.0, abs=1e-9) for r in rows)


def test_article_scores_extracts_only_articles(populated_session: Session) -> None:
    graph = build_graph(populated_session)
    scores = personalized_pagerank(graph).article_scores()
    ids = {a.id for a in populated_session.query(Article).all()}
    assert set(scores) <= ids
    assert scores
