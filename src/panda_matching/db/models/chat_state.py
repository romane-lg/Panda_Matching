from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from panda_matching.db.base import Base


class ChatState(Base):
    __tablename__ = "chat_state"
    __table_args__ = {"schema": "core"}

    session_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("core.chat_sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
