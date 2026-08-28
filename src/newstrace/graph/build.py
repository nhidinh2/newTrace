"""Typed news graph over articles, sources, stories, claims and entities.

Node ids are namespaced strings (``article:12``, ``source:example.com``) so a
single NetworkX graph can hold every node type.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.clustering.labeling import extract_entities
from newstrace.models import Article, Claim, ClaimEvidence, ClusterMembership
from newstrace.utils import ensure_utc, utcnow

NODE_ARTICLE = "article"
NODE_SOURCE = "source"
NODE_STORY = "story"
NODE_CLAIM = "claim"
NODE_ENTITY = "entity"

EDGE_PUBLISHED_BY = "PUBLISHED_BY"
EDGE_BELONGS_TO = "BELONGS_TO"
EDGE_REPORTS = "REPORTS"
EDGE_DISPUTES = "DISPUTES"
EDGE_MENTIONS = "MENTIONS"
EDGE_NEAR_DUPLICATE_OF = "NEAR_DUPLICATE_OF"


def article_node(article_id: int) -> str:
    return f"{NODE_ARTICLE}:{article_id}"


def source_node(domain: str) -> str:
    return f"{NODE_SOURCE}:{domain.lower()}"


def story_node(story_id: int) -> str:
    return f"{NODE_STORY}:{story_id}"


def claim_node(claim_id: int) -> str:
    return f"{NODE_CLAIM}:{claim_id}"


def entity_node(name: str) -> str:
    return f"{NODE_ENTITY}:{name.lower()}"


def parse_node(node: str) -> tuple[str, str]:
    kind, _, value = node.partition(":")
    return kind, value


@dataclass
class GraphStats:
    nodes: int
    edges: int
    node_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]
    connected_components: int
    largest_component_fraction: float
    mean_degree: float
    max_degree: int
    degree_histogram: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "node_type_counts": self.node_type_counts,
            "edge_type_counts": self.edge_type_counts,
            "connected_components": self.connected_components,
            "largest_component_fraction": self.largest_component_fraction,
            "mean_degree": self.mean_degree,
            "max_degree": self.max_degree,
            "degree_histogram": self.degree_histogram,
        }


def build_graph(
    session: Session,
    *,
    articles: Sequence[Article] | None = None,
    include_entities: bool = True,
    include_claims: bool = True,
    max_entities_per_article: int = 8,
    duplicate_edge_weight: float = 0.2,
) -> nx.DiGraph:
    """Build the directed, weighted news graph from persisted state."""
    if articles is None:
        articles = list(session.execute(select(Article)).scalars())
    article_ids = [a.id for a in articles]
    graph = nx.DiGraph()

    for article in articles:
        node = article_node(article.id)
        published = ensure_utc(article.published_at)
        graph.add_node(
            node,
            kind=NODE_ARTICLE,
            title=article.title,
            domain=article.source_domain,
            published_at=published.isoformat() if published else None,
            is_near_duplicate=bool(article.is_near_duplicate),
        )
        src = source_node(article.source_domain or "unknown")
        graph.add_node(src, kind=NODE_SOURCE, domain=article.source_domain)
        graph.add_edge(node, src, kind=EDGE_PUBLISHED_BY, weight=1.0)
        graph.add_edge(src, node, kind=EDGE_PUBLISHED_BY, weight=1.0)

        if article.is_near_duplicate and article.duplicate_of_article_id in set(article_ids):
            other = article_node(article.duplicate_of_article_id)
            graph.add_edge(node, other, kind=EDGE_NEAR_DUPLICATE_OF, weight=duplicate_edge_weight)
            graph.add_edge(other, node, kind=EDGE_NEAR_DUPLICATE_OF, weight=duplicate_edge_weight)

        if include_entities:
            for name in extract_entities(article.body_text)[:max_entities_per_article]:
                ent = entity_node(name)
                graph.add_node(ent, kind=NODE_ENTITY, name=name)
                graph.add_edge(node, ent, kind=EDGE_MENTIONS, weight=1.0)
                graph.add_edge(ent, node, kind=EDGE_MENTIONS, weight=1.0)

    memberships = list(
        session.execute(
            select(ClusterMembership).where(ClusterMembership.article_id.in_(article_ids))
        ).scalars()
    )
    for membership in memberships:
        a = article_node(membership.article_id)
        s = story_node(membership.story_cluster_id)
        graph.add_node(s, kind=NODE_STORY, story_id=membership.story_cluster_id)
        weight = max(0.05, float(membership.similarity))
        graph.add_edge(a, s, kind=EDGE_BELONGS_TO, weight=weight)
        graph.add_edge(s, a, kind=EDGE_BELONGS_TO, weight=weight)

    if include_claims:
        evidence = list(
            session.execute(
                select(ClaimEvidence).where(ClaimEvidence.article_id.in_(article_ids))
            ).scalars()
        )
        claim_ids = {e.claim_id for e in evidence}
        claims = {
            c.id: c for c in session.execute(select(Claim).where(Claim.id.in_(claim_ids))).scalars()
        }
        for ev in evidence:
            claim = claims.get(ev.claim_id)
            if claim is None:
                continue
            c = claim_node(claim.id)
            graph.add_node(c, kind=NODE_CLAIM, text=claim.normalized_claim)
            kind = EDGE_DISPUTES if ev.stance == "disputes" else EDGE_REPORTS
            weight = max(0.05, float(ev.confidence) or 0.5)
            graph.add_edge(article_node(ev.article_id), c, kind=kind, weight=weight)
            graph.add_edge(c, article_node(ev.article_id), kind=kind, weight=weight)

    graph.graph["built_at"] = utcnow().isoformat()
    return graph


def graph_stats(graph: nx.DiGraph) -> GraphStats:
    """Size, degree distribution and component structure of the graph."""
    node_types = Counter(str(data.get("kind", "unknown")) for _, data in graph.nodes(data=True))
    edge_types = Counter(str(data.get("kind", "unknown")) for _, _, data in graph.edges(data=True))
    n = graph.number_of_nodes()
    undirected = graph.to_undirected(as_view=True)
    components = list(nx.connected_components(undirected)) if n else []
    largest = max((len(c) for c in components), default=0)
    degrees = [d for _, d in graph.degree()]
    return GraphStats(
        nodes=n,
        edges=graph.number_of_edges(),
        node_type_counts=dict(sorted(node_types.items())),
        edge_type_counts=dict(sorted(edge_types.items())),
        connected_components=len(components),
        largest_component_fraction=(largest / n) if n else 0.0,
        mean_degree=(sum(degrees) / n) if n else 0.0,
        max_degree=max(degrees, default=0),
        degree_histogram=nx.degree_histogram(graph) if n else [],
    )
