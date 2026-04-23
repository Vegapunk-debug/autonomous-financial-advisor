"""Phase 1: Market Intelligence Layer — trend analysis, sector extraction, news classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MarketSentiment:
    overall: str  # BULLISH / BEARISH / NEUTRAL
    confidence: float  # 0-1
    drivers: list[str] = field(default_factory=list)


@dataclass
class SectorTrend:
    sector: str
    day_change: float
    weekly_change: float | None
    sentiment: str
    trend: str
    drivers: list[str]
    is_rate_sensitive: bool = False
    is_defensive: bool = False


@dataclass
class ClassifiedNews:
    id: str
    headline: str
    sentiment: str
    sentiment_score: float
    scope: str
    impact_level: str
    affected_sectors: list[str]
    affected_stocks: list[str]
    causal_factors: list[str]
    has_conflict: bool = False
    conflict_explanation: str = ""


class MarketIntelligence:
    def __init__(self, market_data: dict, news_data: list[dict],
                 historical: dict, sector_mapping: dict) -> None:
        self._market = market_data
        self._news = news_data
        self._historical = historical
        self._sectors = sector_mapping

    def analyze_market_sentiment(self) -> MarketSentiment: