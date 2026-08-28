"""NewsTrace command line interface."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from newstrace import __version__
from newstrace.config import get_settings, load_topics
from newstrace.db import create_all, session_scope
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.logging import configure_logging
from newstrace.search import SearchFilters, SearchRequest
from newstrace.search import search as run_search
from newstrace.utils import set_global_seed, truncate

app = typer.Typer(
    name="newstrace",
    help="Streaming news story, claim and source-evidence tracker.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(settings.random_seed)
    create_all()


@app.callback()
def main_callback() -> None:
    """Initialise logging, seeds and the database schema."""
    _bootstrap()


@app.command()
def version() -> None:
    """Print the NewsTrace version."""
    console.print(f"NewsTrace {__version__}")


@app.command()
def init_db() -> None:
    """Create database tables directly (Alembic is preferred for real deployments)."""
    create_all()
    console.print("[green]Database schema ready.[/green]")


@app.command()
def topics() -> None:
    """List configured topics."""
    table = Table(title="Configured topics")
    table.add_column("key")
    table.add_column("display name")
    table.add_column("feeds", justify="right")
    table.add_column("GDELT query", overflow="fold")
    for key, topic in load_topics().items():
        table.add_row(key, topic.label(), str(len(topic.feeds)), truncate(topic.gdelt_query, 70))
    console.print(table)


@app.command()
def ingest(
    source: Annotated[str, typer.Option(help="fixtures | gdelt | rss")] = "gdelt",
    topic: Annotated[str | None, typer.Option(help="Restrict to one configured topic")] = None,
    timespan: Annotated[str, typer.Option(help="GDELT timespan, e.g. 24h")] = "24h",
    max_records: Annotated[int, typer.Option(help="GDELT max records (<=250)")] = 250,
    windows: Annotated[
        int, typer.Option(help="Split the GDELT timespan into N sub-windows (>250 records)")
    ] = 1,
    refresh: Annotated[bool, typer.Option(help="Bypass the response cache")] = False,
    index: Annotated[bool, typer.Option(help="Embed and cluster after ingesting")] = True,
) -> None:
    """Ingest articles from fixtures, GDELT or RSS."""
    options = IngestOptions(
        source=source,
        topic=topic,
        timespan=timespan,
        max_records=max_records,
        windows=windows,
        refresh=refresh,
        index=index,
    )
    with session_scope() as session:
        run, result = run_ingestion(session, options)
        console.print(
            f"[bold]Run {run.id}[/bold] ({run.source}) status={run.status} "
            f"fetched={result.fetched} inserted={result.inserted} "
            f"duplicates={result.duplicates} skipped={result.skipped} failures={result.failures}"
        )
        for message in result.errors[:5]:
            console.print(f"  [yellow]{message}[/yellow]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural-language query")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of results")] = 10,
    method: Annotated[
        str, typer.Option(help="full | svd | gaussian_rp | sparse_rp | tfidf")
    ] = "full",
    dimension: Annotated[int | None, typer.Option(help="Compressed dimension")] = None,
    topic: Annotated[str | None, typer.Option(help="Restrict to a topic")] = None,
    domain: Annotated[str | None, typer.Option(help="Restrict to a source domain")] = None,
    pagerank: Annotated[bool, typer.Option(help="Rerank with personalized PageRank")] = False,
    include_duplicates: Annotated[bool, typer.Option(help="Include near-duplicates")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON")] = False,
) -> None:
    """Search stored articles and show why each result ranked where it did."""
    request = SearchRequest(
        query=query,
        filters=SearchFilters(
            topic=topic, source_domain=domain, include_duplicates=include_duplicates
        ),
        method=method,
        dimension=dimension,
        rerank_with_pagerank=pagerank,
        top_k=top_k,
    )
    with session_scope() as session:
        response = run_search(session, request)
        if as_json:
            from newstrace.api.serializers import search_item_to_schema

            console.print_json(
                json.dumps(
                    {
                        "query": response.query,
                        "method": response.method,
                        "took_ms": round(response.took_ms, 2),
                        "results": [
                            search_item_to_schema(i).model_dump(mode="json") for i in response.items
                        ],
                        "notes": response.notes,
                    }
                )
            )
            return
        table = Table(title=f"{response.query}  ({response.method}, {response.took_ms:.1f} ms)")
        table.add_column("#", justify="right")
        table.add_column("score", justify="right")
        table.add_column("title", overflow="fold")
        table.add_column("source")
        table.add_column("published")
        for item in response.items:
            table.add_row(
                str(item.hit.rank),
                f"{item.hit.score:.3f}",
                truncate(item.article.title, 70),
                item.article.source_domain,
                item.article.published_at.strftime("%Y-%m-%d %H:%M")
                if item.article.published_at
                else "—",
            )
        console.print(table)
        for item in response.items[:3]:
            console.print(f"  [dim]{item.explanation}[/dim]")
        for note in response.notes:
            console.print(f"  [yellow]{note}[/yellow]")


@app.command()
def stories(
    limit: Annotated[int, typer.Option(help="Maximum stories to show")] = 15,
    topic: Annotated[str | None, typer.Option(help="Restrict to a topic")] = None,
) -> None:
    """List the current story clusters."""
    from newstrace.stories import list_stories, load_story

    with session_scope() as session:
        table = Table(title="Top stories")
        table.add_column("id", justify="right")
        table.add_column("story", overflow="fold")
        table.add_column("articles", justify="right")
        table.add_column("independent sources", justify="right")
        table.add_column("duplicates", justify="right")
        table.add_column("latest")
        for story in list_stories(session, topic=topic, limit=limit):
            bundle = load_story(session, story.id)
            table.add_row(
                str(story.id),
                truncate(story.display_title, 66),
                str(story.article_count),
                str(story.distinct_domain_count),
                str(len(bundle.duplicates) if bundle else 0),
                story.last_published_at.strftime("%Y-%m-%d %H:%M")
                if story.last_published_at
                else "—",
            )
        console.print(table)


@app.command()
def story(story_id: Annotated[int, typer.Argument(help="Story cluster id")]) -> None:
    """Show a story timeline and its grounded extractive summary."""
    from newstrace.stories import build_timeline, load_story
    from newstrace.summarization.extractive import ExtractiveSummarizer

    with session_scope() as session:
        bundle = load_story(session, story_id)
        if bundle is None:
            console.print(f"[red]Story {story_id} not found.[/red]")
            raise typer.Exit(code=1)
        console.print(f"[bold]{bundle.story.display_title}[/bold]")
        console.print(
            f"{len(bundle.articles)} articles, {len(bundle.independent_domains)} independent "
            f"sources, {len(bundle.duplicates)} near-duplicates\n"
        )
        console.print("[bold]Timeline[/bold]")
        for entry in build_timeline(bundle):
            stamp = entry.published_at.strftime("%Y-%m-%d %H:%M") if entry.published_at else "—"
            marker = " [dim](duplicate)[/dim]" if entry.is_near_duplicate else ""
            first = " [green](earliest available report)[/green]" if entry.is_first_report else ""
            console.print(f"  {stamp}  {entry.article.source_domain}{marker}{first}")
            console.print(f"    {truncate(entry.article.title, 96)}")
            for sentence in entry.new_sentences[:2]:
                console.print(f"    [cyan]new:[/cyan] {truncate(sentence, 96)}")
        summary = ExtractiveSummarizer(session).summarize_bundle(bundle)
        console.print("\n[bold]Grounded summary[/bold]")
        index = summary.evidence_index()
        section_labels = {
            "new_developments": "NEW",
            "repeated_reporting": "REPEATED",
            "single_source_claims": "SINGLE SOURCE",
            "disagreements_or_uncertainty": "DISPUTED",
        }
        for statement in summary.statements:
            # Escape the label: Rich would otherwise read [name] as markup.
            label = section_labels.get(statement.section.value, statement.section.value)
            console.print(f"  [cyan]{label}[/cyan] — {statement.text}")
            for eid in statement.evidence_ids:
                evidence = index[eid]
                console.print(f"      [dim]{evidence.source_domain} — {evidence.url}[/dim]")
        for note in summary.notes:
            console.print(f"  [yellow]{note}[/yellow]")


@app.command()
def experiment(
    methods: Annotated[
        str, typer.Option(help="Comma-separated methods")
    ] = "full,svd,gaussian_rp,sparse_rp",
    dimensions: Annotated[str, typer.Option(help="Comma-separated dimensions")] = "32,64,128,256",
    seed: Annotated[int, typer.Option(help="Random seed")] = 549,
    top_k: Annotated[int, typer.Option(help="Retrieval depth")] = 10,
) -> None:
    """Run the dimensionality-reduction sweep and write an experiment report."""
    from newstrace.evaluation.systems import ExperimentConfig, run_experiment

    config = ExperimentConfig(
        methods=[m.strip() for m in methods.split(",") if m.strip()],
        dimensions=[int(d) for d in dimensions.split(",") if d.strip()],
        seed=seed,
        top_k=top_k,
    )
    with session_scope() as session:
        run = run_experiment(session, config)
        console.print(f"Experiment {run.id}: {run.status}")
        console.print(f"Artifacts: {run.artifact_path}")
        if run.errors:
            console.print(f"[red]{run.errors}[/red]")


@app.command()
def health() -> None:
    """Print system health as JSON."""
    from newstrace.api.routes.health import health as health_route

    with session_scope() as session:
        console.print_json(health_route(session).model_dump_json())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
