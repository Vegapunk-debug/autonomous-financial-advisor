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
        indices = self._market.get("indices", {})
        breadth = self._historical.get("market_breadth", {})
        fii = self._historical.get("fii_dii_data", {})

        bearish_signals = 0
        total = 0
        for idx_data in indices.values():
            total += 1
            if idx_data.get("change_percent", 0) < 0:
                bearish_signals += 1

        # FII flow signal
        fii_net = fii.get("fii", {}).get("net_value_cr", 0)
        if fii_net < -2000:
            bearish_signals += 1
        total += 1

        # Breadth signal
        adr = breadth.get("nifty50", {}).get("advance_decline_ratio", 1)
        if adr < 0.5:
            bearish_signals += 1
        total += 1

        ratio = bearish_signals / max(total, 1)
        if ratio >= 0.6:
            sentiment = "BEARISH"
        elif ratio <= 0.3:
            sentiment = "BULLISH"
        else:
            sentiment = "NEUTRAL"

        drivers = []
        if fii_net < -2000:
            drivers.append(f"FII net selling of Rs{abs(fii_net):,.0f} crore")
        if adr < 0.5:
            n50 = breadth.get("nifty50", {})
            drivers.append(f"Weak breadth: {n50.get('advances', 0)} advances vs {n50.get('declines', 0)} declines")
        # Add index-level drivers
        for name, data in indices.items():
            chg = data.get("change_percent", 0)
            if abs(chg) >= 1.0:
                direction = "fell" if chg < 0 else "rose"
                drivers.append(f"{data.get('name', name)} {direction} {abs(chg):.2f}%")

        # Confidence: higher when signal is more decisive
        if sentiment == "NEUTRAL":
            confidence = 0.5
        else:
            confidence = min(0.95, 0.5 + abs(ratio - 0.5))

        return MarketSentiment(
            overall=sentiment,
            confidence=confidence,
            drivers=drivers,
        )

    def extract_sector_trends(self) -> list[SectorTrend]:
        sector_perf = self._market.get("sector_performance", {})
        weekly = self._historical.get("sector_weekly_performance", {})
        sector_defs = self._sectors.get("sectors", {})
        rate_sensitive = set(self._sectors.get("rate_sensitive_sectors", []))
        defensive = set(self._sectors.get("defensive_sectors", []))

        trends: list[SectorTrend] = []
        for sector_name, perf in sector_perf.items():
            w = weekly.get(sector_name, {})
            info = sector_defs.get(sector_name, {})
            trends.append(SectorTrend(
                sector=sector_name,
                day_change=perf.get("change_percent", 0),
                weekly_change=w.get("weekly_change_percent"),
                sentiment=perf.get("sentiment", "NEUTRAL"),
                trend=w.get("trend", "UNKNOWN"),
                drivers=perf.get("key_drivers", []),
                is_rate_sensitive=sector_name in rate_sensitive,
                is_defensive=sector_name in defensive,
            ))
        # Sort by absolute impact
        trends.sort(key=lambda t: abs(t.day_change), reverse=True)
        return trends

    def classify_news(self) -> list[ClassifiedNews]:
        classified: list[ClassifiedNews] = []
        for n in self._news:
            entities = n.get("entities", {})
            classified.append(ClassifiedNews(
                id=n["id"],
                headline=n["headline"],
                sentiment=n.get("sentiment", "NEUTRAL"),
                sentiment_score=n.get("sentiment_score", 0),
                scope=n.get("scope", "MARKET_WIDE"),
                impact_level=n.get("impact_level", "LOW"),
                affected_sectors=entities.get("sectors", []),
                affected_stocks=entities.get("stocks", []),
                causal_factors=n.get("causal_factors", []),
                has_conflict=n.get("conflict_flag", False),
                conflict_explanation=n.get("conflict_explanation", ""),
            ))
        # Sort: HIGH impact first, then by absolute sentiment score
        priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        classified.sort(key=lambda c: (priority.get(c.impact_level, 0), abs(c.sentiment_score)), reverse=True)
        return classified

    def get_high_impact_news(self) -> list[ClassifiedNews]:
        return [n for n in self.classify_news() if n.impact_level == "HIGH"]

    def get_news_for_stock(self, symbol: str) -> list[ClassifiedNews]:
        return [n for n in self.classify_news() if symbol in n.affected_stocks]

    def get_news_for_sector(self, sector: str) -> list[ClassifiedNews]:
        return [n for n in self.classify_news() if sector in n.affected_sectors]

    def get_conflicting_signals(self) -> list[ClassifiedNews]:
        return [n for n in self.classify_news() if n.has_conflict]

    def build_summary(self) -> dict[str, Any]:
        sentiment = self.analyze_market_sentiment()
        sectors = self.extract_sector_trends()
        high_impact = self.get_high_impact_news()
        conflicts = self.get_conflicting_signals()

        return {
            "market_sentiment": {
                "overall": sentiment.overall,
                "confidence": round(sentiment.confidence, 2),
                "drivers": sentiment.drivers,
            },
            "top_sectors": [
                {"sector": s.sector, "change": s.day_change, "sentiment": s.sentiment,
                 "trend": s.trend, "drivers": s.drivers}
                for s in sectors[:5]
            ],
            "high_impact_news": [
                {"headline": n.headline, "sentiment": n.sentiment, "scope": n.scope,
                 "sectors": n.affected_sectors, "stocks": n.affected_stocks}
                for n in high_impact
            ],
            "conflicting_signals": [
                {"headline": n.headline, "explanation": n.conflict_explanation}
                for n in conflicts
            ],
        }
