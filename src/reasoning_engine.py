"""Phase 3: Autonomous Reasoning Engine — causal linking, conflict resolution, prioritization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .market_intelligence import MarketIntelligence, ClassifiedNews, SectorTrend
from .portfolio_analytics import PortfolioAnalytics, PortfolioSummary, HoldingAnalysis


@dataclass
class CausalChain:
    trigger: str          # The root cause (e.g., "RBI Hawkish Stance")
    chain: list[str]      # Step-by-step causal links
    portfolio_impact: str  # Final impact description
    confidence: float      # 0-1
    affected_holdings: list[str]
    impact_magnitude: float  # portfolio-level % impact


@dataclass
class ConflictAnalysis:
    symbol: str
    positive_signals: list[str]
    negative_signals: list[str]
    resolution: str
    net_assessment: str  # POSITIVE / NEGATIVE / UNCERTAIN


@dataclass
class AdvisorBriefing:
    date: str
    user_name: str
    portfolio_type: str
    market_overview: str
    portfolio_pnl: str
    causal_chains: list[CausalChain]
    conflicts: list[ConflictAnalysis]
    risk_alerts: list[str]
    key_takeaway: str
    confidence_score: float


class ReasoningEngine:
    def __init__(self, market_intel: MarketIntelligence,
                 portfolio_analytics: PortfolioAnalytics, tracer: Any = None) -> None:
        self._intel = market_intel
        self._pa = portfolio_analytics
        self.tracer = tracer

    def generate_briefing(self) -> AdvisorBriefing:
        market_summary = self._intel.build_summary()
        portfolio = self._pa.analyze()

        causal_chains = self._build_causal_chains(market_summary, portfolio)
        conflicts = self._resolve_conflicts(portfolio)
        market_overview = self._narrate_market(market_summary)
        pnl_narrative = self._narrate_pnl(portfolio, causal_chains)
        risk_alerts = [a.message for a in portfolio.risk_alerts]
        takeaway = self._generate_takeaway(portfolio, causal_chains, conflicts)

        # Overall confidence: weighted by chain confidences
        if causal_chains:
            avg_conf = sum(c.confidence for c in causal_chains) / len(causal_chains)
        else:
            avg_conf = 0.5

        return AdvisorBriefing(
            date=self._intel._market.get("metadata", {}).get("date", ""),
            user_name=portfolio.user_name,
            portfolio_type=portfolio.portfolio_type,
            market_overview=market_overview,
            portfolio_pnl=pnl_narrative,
            causal_chains=causal_chains,
            conflicts=conflicts,
            risk_alerts=risk_alerts,
            key_takeaway=takeaway,
            confidence_score=round(avg_conf, 2),
        )
