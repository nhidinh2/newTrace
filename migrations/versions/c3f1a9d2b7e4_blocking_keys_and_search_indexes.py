"""blocking keys, covering indexes and the full-text index

Three things that were linear scans become index lookups:

* duplicate detection compared each incoming article against a 96-hour window
  of articles capped at 2,000 rows -- ``article_signatures`` holds the SimHash
  bands and title prefix tokens it now blocks on;
* the default search filter ("everything that is not a near-duplicate") read
  every article row to collect ids -- ``ix_articles_live`` covers it;
* the retrieval cache re-checks ``(count, max id)`` per representation before
  every query -- ``ix_embedding_lookup`` makes that index-only.

``articles_fts`` is an FTS5 virtual table; it is created with raw SQL because
it has no ORM model, and it is skipped when the SQLite build lacks FTS5.

Revision ID: c3f1a9d2b7e4
Revises: b79054f4422e
Create Date: 2026-09-21 21:15:00.000000
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3f1a9d2b7e4"
down_revision: str | None = "b79054f4422e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
USING fts5(title, body, tokenize = 'unicode61 remove_diacritics 2')
"""


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    op.create_table(
        "article_signatures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("article_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("value", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(["article_id"], ["articles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("article_id", "kind", "value", name="uq_article_signature"),
    )
    with op.batch_alter_table("article_signatures", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_article_signatures_article_id"), ["article_id"], unique=False
        )
        batch_op.create_index(
            "ix_article_signatures_lookup", ["kind", "value", "article_id"], unique=False
        )

    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("simhash", sa.BigInteger(), nullable=True))
        batch_op.create_index(
            "ix_articles_live",
            ["is_near_duplicate", "published_at", "topic", "source_domain", "id"],
            unique=False,
        )

    with op.batch_alter_table("embedding_records", schema=None) as batch_op:
        batch_op.create_index(
            "ix_embedding_lookup", ["method", "dimension", "fit_version", "id"], unique=False
        )

    if _is_sqlite():
        # A SQLite build without FTS5 still gets every other index; search
        # falls back to the Python paths.
        with contextlib.suppress(sa.exc.OperationalError):
            op.execute(FTS_DDL)


def downgrade() -> None:
    if _is_sqlite():
        op.execute("DROP TABLE IF EXISTS articles_fts")

    with op.batch_alter_table("embedding_records", schema=None) as batch_op:
        batch_op.drop_index("ix_embedding_lookup")

    with op.batch_alter_table("articles", schema=None) as batch_op:
        batch_op.drop_index("ix_articles_live")
        batch_op.drop_column("simhash")

    with op.batch_alter_table("article_signatures", schema=None) as batch_op:
        batch_op.drop_index("ix_article_signatures_lookup")
        batch_op.drop_index(batch_op.f("ix_article_signatures_article_id"))
    op.drop_table("article_signatures")
