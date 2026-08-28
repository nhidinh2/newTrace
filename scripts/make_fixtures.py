"""Generate the committed synthetic fixtures and their story labels.

Everything produced here is invented: the publishers are ``*.example`` domains,
the headlines and sentences are written for this repository, and no real article
text is copied.  That keeps the demo runnable offline without redistributing
anyone's copyrighted reporting.

Run: ``python scripts/make_fixtures.py``
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures"
LABELS = ROOT / "data" / "labels"
TEST_FIXTURES = ROOT / "tests" / "fixtures"

BASE = datetime(2026, 8, 18, 6, 0, tzinfo=UTC)


@dataclass
class Report:
    domain: str
    title: str
    description: str
    offset_hours: float
    duplicate_of: int | None = None
    slug: str = ""


@dataclass
class Story:
    label: str
    topic: str
    reports: list[Report] = field(default_factory=list)


STORIES: list[Story] = [
    Story(
        label="story_agent_benchmark",
        topic="ai_models",
        reports=[
            Report(
                "techledger.example",
                "Northwind Labs releases Atlas 3 agent model with 128k context window",
                "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark "
                "and runs with a 128,000 token context window. The company said the model is "
                "available to developers on Tuesday. Pricing was not announced.",
                0,
            ),
            Report(
                "wirefront.example",
                "Atlas 3 launch: Northwind claims 71.4 percent on agent benchmark",
                "Northwind Labs announced Atlas 3 on Tuesday. The company said the model reached "
                "71.4 percent on the public agent benchmark. Independent evaluations of the model "
                "have not yet been published.",
                1.5,
            ),
            Report(
                "circuitdesk.example",
                "Developers get access to Northwind's Atlas 3 agent model",
                "Atlas 3 became available to developers on Tuesday, Northwind Labs said. Two "
                "researchers who tested an early build said latency improved noticeably over the "
                "previous release. Northwind declined to share training data details.",
                3.0,
            ),
            Report(
                "syndicatewire.example",
                "Northwind Labs releases Atlas 3 agent model with 128k context window",
                "Northwind Labs said Atlas 3 scores 71.4 percent on the public agent benchmark "
                "and runs with a 128,000 token context window. The company said the model is "
                "available to developers on Tuesday. Pricing was not announced.",
                4.0,
                duplicate_of=0,
            ),
            Report(
                "modelbeat.example",
                "Northwind's Atlas 3 agent benchmark score disputed by analysts",
                "Two analysts said Atlas 3 does not score 71.4 percent on the public agent "
                "benchmark, putting the comparable figure closer to 64 percent. Northwind Labs "
                "said its methodology is documented in the Atlas 3 model card.",
                9.0,
            ),
        ],
    ),
    Story(
        label="story_eu_ai_rules",
        topic="ai_safety",
        reports=[
            Report(
                "policybench.example",
                "EU regulators publish draft transparency rules for general-purpose AI models",
                "The draft rules would require providers of general-purpose models to publish "
                "training-data summaries and incident reports. Regulators said the consultation "
                "closes on 30 September. Industry groups have asked for a longer comment window.",
                2.0,
            ),
            Report(
                "civicsignal.example",
                "Draft EU rules would force AI providers to publish training-data summaries",
                "Under the draft published this week, providers of general-purpose AI models "
                "would file incident reports and publish training-data summaries. The "
                "consultation closes on 30 September, regulators said.",
                4.5,
            ),
            Report(
                "wirefront.example",
                "Industry groups push back on EU AI transparency consultation timeline",
                "Three industry associations said the 30 September deadline gives too little time "
                "to respond. A regulator spokesperson said the timeline is not expected to change. "
                "The draft does not set penalties yet.",
                11.0,
            ),
            Report(
                "techledger.example",
                "What the EU's draft general-purpose AI rules would require",
                "The draft covers training-data summaries, incident reporting and downstream "
                "documentation. Regulators said enforcement details will follow in a later text. "
                "No penalty schedule has been published.",
                26.0,
            ),
        ],
    ),
    Story(
        label="story_chip_export",
        topic="chips_and_compute",
        reports=[
            Report(
                "chipwatch.example",
                "Meridian Semiconductor to build $4.2 billion packaging plant in Arizona",
                "Meridian Semiconductor said the plant will cost $4.2 billion and employ 1,800 "
                "people when it opens in 2029. Construction begins in the first quarter. State "
                "officials said incentives were part of the agreement.",
                5.0,
            ),
            Report(
                "circuitdesk.example",
                "Meridian confirms Arizona advanced packaging site, 1,800 jobs promised",
                "The company confirmed the Arizona site on Wednesday and said it expects 1,800 "
                "jobs. Meridian put the investment at $4.2 billion. The facility is scheduled to "
                "open in 2029.",
                6.5,
            ),
            Report(
                "marketfloor.example",
                "Meridian's Arizona packaging plant listed at $3.8 billion in state filing",
                "A state filing reviewed on Thursday lists Meridian Semiconductor's Arizona "
                "packaging plant at $3.8 billion, not the $4.2 billion the company announced. "
                "Meridian did not respond to a request for comment on the discrepancy.",
                20.0,
            ),
            Report(
                "syndicatewire.example",
                "Meridian Semiconductor to build $4.2 billion packaging plant in Arizona",
                "Meridian Semiconductor said the plant will cost $4.2 billion and employ 1,800 "
                "people when it opens in 2029. Construction begins in the first quarter. State "
                "officials said incentives were part of the agreement.",
                7.0,
                duplicate_of=0,
            ),
        ],
    ),
    Story(
        label="story_prompt_injection",
        topic="ai_safety",
        reports=[
            Report(
                "secureloop.example",
                "Researchers disclose prompt-injection flaw affecting three agent frameworks",
                "Researchers at Halden University disclosed a prompt-injection technique that "
                "bypassed tool-use restrictions in three open-source agent frameworks. Two "
                "maintainers shipped patches on Monday. The third has not responded publicly.",
                8.0,
            ),
            Report(
                "modelbeat.example",
                "Agent frameworks patched after Halden prompt-injection disclosure",
                "Two of the three affected agent frameworks released patches on Monday following "
                "the Halden University disclosure. Researchers said the technique required no "
                "special access to the target system.",
                10.5,
            ),
            Report(
                "policybench.example",
                "Halden disclosure renews debate over agent tool permissions",
                "Security engineers said the disclosure shows tool permissions are still granted "
                "too broadly by default. One maintainer said the reported technique does not work "
                "against the framework's current release.",
                28.0,
            ),
        ],
    ),
    Story(
        label="story_datacenter_power",
        topic="chips_and_compute",
        reports=[
            Report(
                "marketfloor.example",
                "Grid operator warns data-centre demand could add 6 gigawatts by 2030",
                "The regional grid operator said data-centre load could add 6 gigawatts of demand "
                "by 2030 under its high-growth scenario. The operator said transmission upgrades "
                "would be needed. No decision has been taken on cost allocation.",
                13.0,
            ),
            Report(
                "chipwatch.example",
                "Data-centre growth pushes regional grid planners toward transmission upgrades",
                "Planners said the 6 gigawatt high-growth figure assumes current expansion "
                "announcements are completed on schedule. Utilities have asked regulators to "
                "clarify who pays for the upgrades.",
                15.0,
            ),
        ],
    ),
    Story(
        label="story_open_weights",
        topic="ai_models",
        reports=[
            Report(
                "modelbeat.example",
                "Kestrel Institute publishes open-weight 12B model under a research licence",
                "The Kestrel Institute released a 12-billion-parameter model with open weights "
                "under a research-only licence. The institute said commercial use requires a "
                "separate agreement. Evaluation results were published alongside the release.",
                17.0,
            ),
            Report(
                "techledger.example",
                "Kestrel's 12B open-weight release comes with research-only terms",
                "Kestrel's release includes weights and evaluation results but restricts "
                "commercial use. Two developers said the licence would keep them from deploying "
                "the model in products.",
                19.0,
            ),
            Report(
                "wirefront.example",
                "Open-weight model releases accelerate as Kestrel joins with 12B system",
                "Kestrel Institute's release is the fourth open-weight model published this month. "
                "The institute said it trained the system on 1.4 trillion tokens.",
                22.0,
            ),
        ],
    ),
    Story(
        label="story_ai_hiring",
        topic="ai_models",
        reports=[
            Report(
                "civicsignal.example",
                "Labour board opens inquiry into automated screening at three employers",
                "The labour board said it opened an inquiry into automated candidate screening at "
                "three employers after complaints were filed in June. The employers said their "
                "systems are reviewed by human recruiters.",
                30.0,
            ),
            Report(
                "policybench.example",
                "Employers defend automated screening as labour board inquiry begins",
                "Two of the three employers named in the inquiry said human recruiters review "
                "every rejection. The labour board declined to say when it expects to conclude.",
                33.0,
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Seeded background corpus
#
# The hand-written stories above are small. To make the dimension sweep and the
# clustering metrics meaningful we also generate a larger synthetic corpus with
# a fixed seed. Each background story gets its own fictional organisation,
# figures and dates, and each outlet phrases the event differently, so the
# corpus contains genuine within-story variation rather than copied text.
# ---------------------------------------------------------------------------

ORGS = [
    "Vantara Systems",
    "Blue Harbor AI",
    "Corvid Compute",
    "Larkspur Robotics",
    "Ostrom Analytics",
    "Pellucid Labs",
    "Quillon Networks",
    "Redshift Foundry",
    "Sableport Semiconductors",
    "Tessellate AI",
    "Umbra Research",
    "Verity Compute",
    "Windrow Institute",
    "Yarrow Dynamics",
    "Zephyr Silicon",
    "Ashgrove Data",
    "Bellweather Models",
    "Cinderpeak Cloud",
    "Dunmore Chips",
    "Eastvale AI",
]

BG_DOMAINS = [
    "techledger.example",
    "wirefront.example",
    "circuitdesk.example",
    "modelbeat.example",
    "chipwatch.example",
    "marketfloor.example",
    "policybench.example",
    "secureloop.example",
    "civicsignal.example",
    "gridnotes.example",
    "silicondaily.example",
    "openweights.example",
]

BG_TOPICS = ["ai_models", "ai_safety", "chips_and_compute"]

EVENTS = [
    {
        "topic": "ai_models",
        "headline": "{org} releases {product}, a {size}B-parameter model for {use}",
        "angles": [
            "{org} released {product} on {day}. The company said the {size} billion parameter "
            "model targets {use} and reached {score} percent on its internal evaluation suite. "
            "Availability outside the developer preview was not announced.",
            "{product}, announced by {org} on {day}, is aimed at {use}. {org} reported a {score} "
            "percent internal evaluation score for the {size} billion parameter system. "
            "Independent results are not yet available.",
            "Developers can request access to {product}, {org} said on {day}. The company put the "
            "model at {size} billion parameters and cited a {score} percent score on its own "
            "evaluation suite. Two testers said throughput was the notable change.",
            "{org} said {product} will serve {use} workloads. The {size} billion parameter model "
            "posted {score} percent internally, according to the company. A researcher who saw an "
            "early build said the evaluation configuration was not published.",
        ],
    },
    {
        "topic": "ai_safety",
        "headline": "{org} reports {count} security findings in {product} evaluation",
        "angles": [
            "{org} published {count} security findings from its evaluation of {product} on {day}. "
            "The report said {fixed} findings were fixed before publication. The remaining issues "
            "are under review.",
            "An evaluation released by {org} on {day} lists {count} security findings for "
            "{product}. {org} said {fixed} were resolved before the report went out. No exploit "
            "code was published.",
            "{org}'s {day} report on {product} describes {count} findings, of which {fixed} were "
            "already patched. A maintainer said the remaining findings require configuration "
            "changes rather than code fixes.",
            "Security researchers at {org} said {product} had {count} findings in their review "
            "published {day}. {fixed} were fixed ahead of disclosure. The severity ratings were "
            "assigned by {org} itself.",
        ],
    },
    {
        "topic": "chips_and_compute",
        "headline": "{org} commits ${money} billion to {facility} expansion",
        "angles": [
            "{org} said on {day} it will spend ${money} billion expanding its {facility}. The "
            "company expects the work to finish in {year} and to add {jobs} positions.",
            "A ${money} billion expansion of {org}'s {facility} was confirmed on {day}. {org} said "
            "{jobs} jobs would follow and gave {year} as the completion target.",
            "{org} put the cost of its {facility} expansion at ${money} billion when it announced "
            "the project on {day}. Local officials said {jobs} jobs are expected by {year}.",
            "The {facility} expansion announced by {org} on {day} carries a ${money} billion price "
            "tag. {org} said hiring for the {jobs} new roles begins before {year}.",
        ],
    },
    {
        "topic": "chips_and_compute",
        "headline": "Regulators review {org}'s {money} billion {facility} agreement",
        "angles": [
            "Regulators opened a review of {org}'s ${money} billion {facility} agreement on {day}. "
            "The agency said it expects to decide within {count} months.",
            "{org}'s ${money} billion {facility} deal is under regulatory review as of {day}. A "
            "spokesperson said the company is cooperating and expects clearance.",
            "A review of the ${money} billion {facility} agreement involving {org} began {day}, "
            "the agency said. Two competitors filed objections during the comment period.",
        ],
    },
    {
        "topic": "ai_models",
        "headline": "{org} opens {product} weights to researchers under {licence} terms",
        "angles": [
            "{org} published the weights of {product} on {day} under a {licence} licence. The "
            "institute said it trained the model on {money} trillion tokens.",
            "Researchers can now download {product}, {org} said {day}. The {licence} licence "
            "restricts redistribution. Training used {money} trillion tokens, according to the "
            "release notes.",
            "{org}'s {day} release puts {product} weights under {licence} terms. Two developers "
            "said the licence conditions rule out production use.",
        ],
    },
]

PRODUCTS = [
    "Aster",
    "Basalt",
    "Cobalt",
    "Drift",
    "Ember",
    "Fathom",
    "Glint",
    "Harrow",
    "Ingot",
    "Juniper",
    "Kelvin",
    "Lattice",
    "Mesa",
    "Nimbus",
    "Onyx",
    "Prism",
]
USES = [
    "code generation",
    "document analysis",
    "agent orchestration",
    "retrieval workloads",
    "customer support automation",
    "scientific literature search",
]
FACILITIES = [
    "Fairhaven packaging site",
    "Northgate fabrication plant",
    "Sandhill test facility",
    "Lakemoor assembly line",
    "Riverbend substrate works",
]
LICENCES = ["research-only", "non-commercial", "source-available", "community"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def background_stories(count: int = 34, seed: int = 549) -> list[Story]:
    """Deterministically generate additional labelled stories."""
    rng = random.Random(seed)
    stories: list[Story] = []
    for i in range(count):
        event = EVENTS[i % len(EVENTS)]
        org = ORGS[i % len(ORGS)]
        fields = {
            "org": org,
            "product": f"{PRODUCTS[i % len(PRODUCTS)]} {rng.randint(1, 4)}",
            "size": rng.choice([7, 8, 12, 27, 34, 70]),
            "use": rng.choice(USES),
            "score": round(rng.uniform(48.0, 88.0), 1),
            "count": rng.randint(3, 19),
            "fixed": rng.randint(1, 3),
            "money": round(rng.uniform(0.4, 9.5), 1),
            "jobs": rng.choice([320, 640, 900, 1200, 2100]),
            "year": rng.choice([2028, 2029, 2030, 2031]),
            "facility": rng.choice(FACILITIES),
            "licence": rng.choice(LICENCES),
            "day": rng.choice(DAYS),
        }
        headline = event["headline"].format(**fields)
        angles = list(event["angles"])
        rng.shuffle(angles)
        n_reports = rng.randint(2, 4)
        domains = rng.sample(BG_DOMAINS, n_reports)
        reports: list[Report] = []
        start = 36.0 + i * 3.5
        for j in range(n_reports):
            variant_headline = headline if j == 0 else _vary_headline(headline, fields, j, rng)
            reports.append(
                Report(
                    domain=domains[j],
                    title=variant_headline,
                    description=angles[j % len(angles)].format(**fields),
                    offset_hours=start + j * rng.uniform(0.7, 5.0),
                )
            )
        # One in five background stories is syndicated verbatim by a wire service.
        if i % 5 == 0:
            reports.append(
                Report(
                    domain="syndicatewire.example",
                    title=reports[0].title,
                    description=reports[0].description,
                    offset_hours=start + 6.0,
                    duplicate_of=0,
                )
            )
        stories.append(Story(label=f"bg_story_{i:03d}", topic=event["topic"], reports=reports))
    return stories


def _vary_headline(headline: str, fields: dict, j: int, rng: random.Random) -> str:
    """Rewrite a headline the way a different outlet would, without copying it."""
    org = fields["org"]
    tail = headline.split(" ", 1)[1] if " " in headline else headline
    patterns = [
        f"{tail.capitalize()}, {org} says",
        f"What {org} announced: {tail}",
        f"{org}: {tail}",
        f"Report — {tail}",
    ]
    return patterns[(j - 1) % len(patterns)]


def canonical(domain: str, slug: str) -> str:
    return f"https://{domain}/{slug}"


def slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in title]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:70]


def build() -> tuple[
    dict[str, list[dict[str, object]]], list[dict[str, str]], list[dict[str, str]]
]:
    """Return per-topic GDELT payloads, RSS rows and label rows."""
    gdelt: dict[str, list[dict[str, object]]] = {}
    rss_rows: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []

    for story in [*STORIES, *background_stories()]:
        for report in story.reports:
            slug = report.slug or slugify(report.title)
            url = canonical(report.domain, slug)
            seen = (BASE + timedelta(hours=report.offset_hours)).strftime("%Y%m%dT%H%M%SZ")
            record = {
                "url": url,
                "url_mobile": "",
                "title": report.title,
                "seendate": seen,
                "socialimage": "",
                "domain": report.domain,
                "language": "English",
                "sourcecountry": "United States",
            }
            gdelt.setdefault(story.topic, []).append(record)
            # The feed carries the description text that GDELT's artlist mode omits.
            if True:
                rss_rows.append(
                    {
                        "topic": story.topic,
                        "url": url,
                        "title": report.title,
                        "description": report.description,
                        "published": (BASE + timedelta(hours=report.offset_hours)).strftime(
                            "%a, %d %b %Y %H:%M:%S +0000"
                        ),
                        "domain": report.domain,
                    }
                )
            labels.append({"canonical_url": url, "story_label": story.label, "topic": story.topic})
    return gdelt, rss_rows, labels


def write_rss(path: Path, rows: list[dict[str, str]], title: str) -> None:
    items = "\n".join(
        f"""    <item>
      <title>{escape(row["title"])}</title>
      <link>{escape(row["url"])}</link>
      <guid isPermaLink="true">{escape(row["url"])}</guid>
      <description>{escape(row["description"])}</description>
      <pubDate>{row["published"]}</pubDate>
    </item>"""
        for row in rows
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{escape(title)}</title>
    <link>https://fixtures.example/{escape(title)}</link>
    <description>Synthetic NewsTrace fixture feed. All content is invented.</description>
    <language>en-us</language>
{items}
  </channel>
</rss>
"""
    path.write_text(xml, encoding="utf-8")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    TEST_FIXTURES.mkdir(parents=True, exist_ok=True)

    gdelt, rss_rows, labels = build()

    for topic, records in gdelt.items():
        payload = {"articles": records}
        (FIXTURES / f"{topic}.gdelt.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    by_topic: dict[str, list[dict[str, str]]] = {}
    for row in rss_rows:
        by_topic.setdefault(row["topic"], []).append(row)
    for topic, rows in by_topic.items():
        write_rss(FIXTURES / f"{topic}.rss.xml", rows, f"NewsTrace fixture: {topic}")

    with (LABELS / "fixture_story_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["canonical_url", "story_label", "topic"])
        writer.writeheader()
        writer.writerows(labels)

    # Small deterministic fixtures for the unit tests.
    sample_topic = "ai_models"
    (TEST_FIXTURES / "gdelt_sample.json").write_text(
        json.dumps({"articles": gdelt[sample_topic][:4]}, indent=2) + "\n", encoding="utf-8"
    )
    write_rss(
        TEST_FIXTURES / "rss_sample.xml",
        by_topic[sample_topic][:3],
        "NewsTrace test fixture",
    )
    (TEST_FIXTURES / "gdelt_malformed.json").write_text(
        json.dumps({"articles": [{"title": "no url here"}, "not-an-object", {"url": ""}]}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    total = sum(len(v) for v in gdelt.values())
    print(f"Wrote {total} synthetic articles across {len(gdelt)} topics to {FIXTURES}")
    print(f"Wrote {len(labels)} story labels to {LABELS / 'fixture_story_labels.csv'}")


if __name__ == "__main__":
    main()
