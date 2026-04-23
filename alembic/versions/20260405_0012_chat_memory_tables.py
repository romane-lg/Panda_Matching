"""add persistent chat memory tables

Revision ID: 20260405_0012
Revises: 20260405_0011
Create Date: 2026-04-05 19:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260405_0012"
down_revision = "20260405_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_chat_sessions"),
        schema="core",
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=64), nullable=True),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["core.chat_sessions.session_id"],
            name="fk_chat_messages_session_id_chat_sessions",
            ondelete="CASCADE",
        ),
        schema="core",
    )
    op.create_index(
        "ix_core_chat_messages_session_created",
        "chat_messages",
        ["session_id", "created_at"],
        unique=False,
        schema="core",
    )

    op.create_table(
        "chat_state",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column(
            "state_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["core.chat_sessions.session_id"],
            name="fk_chat_state_session_id_chat_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", name="pk_chat_state"),
        schema="core",
    )


def downgrade() -> None:
    op.drop_table("chat_state", schema="core")
    op.drop_index(
        "ix_core_chat_messages_session_created",
        table_name="chat_messages",
        schema="core",
    )
    op.drop_table("chat_messages", schema="core")
    op.drop_table("chat_sessions", schema="core")
