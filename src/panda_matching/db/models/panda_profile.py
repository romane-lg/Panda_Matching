from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from panda_matching.db.base import Base


class PandaProfile(Base):
    __tablename__ = "panda_profiles"
    __table_args__ = {"schema": "core"}

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    chinese_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sex: Mapped[str | None] = mapped_column(String(32), nullable=True)
    age_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_location: Mapped[str | None] = mapped_column(Text, nullable=True)
    zoo_or_facility: Mapped[str | None] = mapped_column(String(255), nullable=True)
    city_region: Mapped[str | None] = mapped_column(String(255), nullable=True)
    country: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mother: Mapped[str | None] = mapped_column(String(255), nullable=True)
    father: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    life_journey: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_twin: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    personality_tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    breeding_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    health_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    babies_had_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lineage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    on_loan: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ownership_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
