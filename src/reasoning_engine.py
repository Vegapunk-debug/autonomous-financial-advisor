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

    def _build_causal_chains(self, market: dict, portfolio: PortfolioSummary) -> list[CausalChain]:
        chains: list[CausalChain] = []
        news_items = self._intel.classify_news()
        sectors_hit = self._intel.extract_sector_trends()
        sector_map = {s.sector: s for s in sectors_hit}

        # Group holdings by sector
        sector_holdings: dict[str, list[HoldingAnalysis]] = {}
        for h in portfolio.holdings:
            if h.asset_type == "STOCK":
                sector_holdings.setdefault(h.sector, []).append(h)

        # For HIGH impact news + MEDIUM stock-specific news hitting portfolio holdings
        portfolio_symbols = {h.symbol for h in portfolio.holdings if h.asset_type == "STOCK"}
        relevant_news = [
            n for n in news_items
            if n.impact_level == "HIGH"
            or (n.impact_level == "MEDIUM" and (
                set(n.affected_stocks) & portfolio_symbols
                or set(n.affected_sectors) & set(sector_holdings.keys())
            ))
        ]

        for news in relevant_news:
            for sector in news.affected_sectors:
                if sector not in sector_holdings and not news.affected_stocks:
                    continue
                sector_trend = sector_map.get(sector)
                if not sector_trend:
                    continue

                affected = []
                total_impact = 0.0
                for h in sector_holdings.get(sector, []):
                    affected.append(h.symbol)
                    total_impact += h.contribution_to_day_pnl

                # Also check stock-specific news
                for stock_sym in news.affected_stocks:
                    for h in portfolio.holdings:
                        if h.symbol == stock_sym and h.symbol not in affected:
                            affected.append(h.symbol)
                            total_impact += h.contribution_to_day_pnl

                if not affected:
                    continue

                chain_steps = [
                    news.headline,
                    f"{sector} sector moved {sector_trend.day_change:+.2f}% (weekly: {sector_trend.weekly_change:+.2f}%)" if sector_trend.weekly_change else f"{sector} sector moved {sector_trend.day_change:+.2f}%",
                ]
                for sym in affected:
                    h = next((x for x in portfolio.holdings if x.symbol == sym), None)
                    if h:
                        chain_steps.append(
                            f"{sym} moved {h.day_change_pct:+.2f}% (weight: {h.weight:.1f}%)"
                        )

                impact_desc = f"Portfolio impact from this chain: {total_impact:+.3f}%"
                confidence = self._calc_chain_confidence(news, sector_trend, affected, portfolio)

                chains.append(CausalChain(
                    trigger=news.headline[:80],
                    chain=chain_steps,
                    portfolio_impact=impact_desc,
                    confidence=confidence,
                    affected_holdings=affected,
                    impact_magnitude=abs(total_impact),
                ))

        # Deduplicate chains with overlapping holdings, keep highest confidence
        chains = self._deduplicate_chains(chains)
        chains.sort(key=lambda c: c.impact_magnitude, reverse=True)
        return chains[:5]  # Top 5 most impactful

    def _calc_chain_confidence(self, news: ClassifiedNews, sector: SectorTrend,
                               holdings: list[str], portfolio: PortfolioSummary) -> float:
        """Score how confident we are in this causal link."""
        conf = 0.5

        # News sentiment aligns with price movement
        if (news.sentiment_score < 0 and sector.day_change < 0) or \
           (news.sentiment_score > 0 and sector.day_change > 0):
            conf += 0.2
        else:
            conf -= 0.15  # Conflict reduces confidence

        # HIGH impact news more reliable
        if news.impact_level == "HIGH":
            conf += 0.1

        # Scope matches — MARKET_WIDE news explaining market-wide moves = higher conf
        if news.scope == "STOCK_SPECIFIC" and len(holdings) == 1:
            conf += 0.1
        elif news.scope == "SECTOR_SPECIFIC":
            conf += 0.05

        # Weekly trend alignment
        if sector.weekly_change is not None:
            if (sector.weekly_change < 0 and sector.day_change < 0) or \
               (sector.weekly_change > 0 and sector.day_change > 0):
                conf += 0.05

        return min(max(conf, 0.1), 0.95)

    def _deduplicate_chains(self, chains: list[CausalChain]) -> list[CausalChain]:
        seen_holdings: set[str] = set()
        unique: list[CausalChain] = []
        # Sort by confidence desc first
        chains.sort(key=lambda c: c.confidence, reverse=True)
        for chain in chains:
            key = frozenset(chain.affected_holdings)
            overlap = key & seen_holdings
            if len(overlap) < len(key) * 0.5:  # Less than 50% overlap
                unique.append(chain)
                seen_holdings.update(key)
        return unique

    def _resolve_conflicts(self, portfolio: PortfolioSummary) -> list[ConflictAnalysis]:
        conflicts: list[ConflictAnalysis] = []
        conflict_news = self._intel.get_conflicting_signals()

        for cn in conflict_news:
            for sym in cn.affected_stocks:
                # Check if this stock is in portfolio
                holding = next((h for h in portfolio.holdings if h.symbol == sym), None)
                if not holding:
                    continue

                stock_news = self._intel.get_news_for_stock(sym)
                pos = [n.headline for n in stock_news if n.sentiment_score > 0]
                neg = [n.headline for n in stock_news if n.sentiment_score < 0]

                # Determine resolution
                if holding.day_change_pct < 0 and pos:
                    resolution = (
                        f"{sym} fell {abs(holding.day_change_pct):.2f}% despite positive company news. "
                        f"Sector-level headwinds (macro/FII selling) are overpowering stock-specific positives."
                    )
                    net = "NEGATIVE"
                elif holding.day_change_pct > 0 and neg:
                    resolution = (
                        f"{sym} rose {holding.day_change_pct:.2f}% despite negative news. "
                        f"Defensive buying or company-specific strength outweighing sector weakness."
                    )
                    net = "POSITIVE"
                else:
                    resolution = f"Mixed signals for {sym} — monitor for clarity."
                    net = "UNCERTAIN"

                conflicts.append(ConflictAnalysis(
                    symbol=sym,
                    positive_signals=pos,
                    negative_signals=neg,
                    resolution=resolution,
                    net_assessment=net,
                ))

        return conflicts

    def _narrate_market(self, summary: dict) -> str:
        s = summary["market_sentiment"]
        parts = [f"Market sentiment: {s['overall']} (confidence: {s['confidence']:.0%})"]
        for d in s["drivers"][:3]:
            parts.append(f"  - {d}")

        parts.append("\nTop sector moves:")
        for sec in summary["top_sectors"][:5]:
            parts.append(f"  {sec['sector']}: {sec['change']:+.2f}% [{sec['sentiment']}]")

        return "\n".join(parts)

    def _narrate_pnl(self, portfolio: PortfolioSummary, chains: list[CausalChain]) -> str:
        parts = [
            f"Portfolio P&L: Rs{portfolio.day_pnl_abs:,.0f} ({portfolio.day_pnl_pct:+.2f}%)",
            f"Total value: Rs{portfolio.total_value:,.0f}",
        ]

        if portfolio.top_gainer:
            g = portfolio.top_gainer
            parts.append(f"Top gainer: {g.symbol} {g.day_change_pct:+.2f}%")
        if portfolio.top_loser:
            l = portfolio.top_loser
            parts.append(f"Top loser: {l.symbol} {l.day_change_pct:+.2f}%")

        if chains:
            parts.append("\nPrimary drivers:")
            for c in chains[:3]:
                parts.append(f"  -> {c.trigger}")
                parts.append(f"     {c.portfolio_impact}")

        return "\n".join(parts)

    def _generate_takeaway(self, portfolio: PortfolioSummary,
                           chains: list[CausalChain],
                           conflicts: list[ConflictAnalysis]) -> str:
        parts = []

        # Main impact
        if chains:
            top = chains[0]
            parts.append(
                f"Your portfolio moved {portfolio.day_pnl_pct:+.2f}% today, "
                f"primarily driven by: {top.trigger}."
            )
            if top.affected_holdings:
                parts.append(
                    f"Most affected holdings: {', '.join(top.affected_holdings[:3])}."
                )

        # Risk alerts
        critical = [a for a in portfolio.risk_alerts if a.level == "CRITICAL"]
        if critical:
            parts.append(f"\nRISK ALERT: {critical[0].message}")

        # Conflicts
        if conflicts:
            parts.append(f"\nNote: {len(conflicts)} holding(s) show conflicting signals — "
                        "price action diverges from news sentiment. Exercise caution.")

        return " ".join(parts) if parts else "No significant signals detected today."

    def generate_narrative(self) -> str:
        """Generate a full natural-language advisory briefing using Groq LLM."""
        market_summary = self._intel.build_summary()
        portfolio = self._pa.analyze()
        chains = self._build_causal_chains(market_summary, portfolio)
        conflicts = self._resolve_conflicts(portfolio)
        
        # Prepare context for LLM
        prompt = f"""
You are an expert Autonomous Financial Advisor.
Your job is to write a comprehensive natural-language advisory narrative.

Market Summary:
{market_summary}

Portfolio Day P&L: {portfolio.day_pnl_pct:+.2f}% (Rs{portfolio.day_pnl_abs:+,.0f})
Portfolio Top Gainer: {portfolio.top_gainer.symbol if portfolio.top_gainer else 'N/A'}
Portfolio Top Loser: {portfolio.top_loser.symbol if portfolio.top_loser else 'N/A'}

Identified Causal Links (Drivers of performance):
{chains}

Identified Conflicts (News vs Price action):
{conflicts}

Write a highly professional, causal, and concise briefing (3-4 paragraphs) that explains WHY the portfolio moved today.
Link Macro News -> Sector Trends -> Individual Stock Performance -> User Portfolio Impact.
If there are conflicts, explain the ambiguity.
Do NOT just list the data. Reason through it like a human advisor. Focus on high-impact signals.
"""
        import os
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not found. Please set it to enable LLM reasoning."
        
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
            )
            narrative = response.choices[0].message.content.strip()
            
            if hasattr(self, 'tracer') and self.tracer:
                usage = {
                    "prompt": response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else 0,
                    "completion": response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0,
                    "total": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0,
                }
                self.tracer.log_llm_call(
                    name="generate_narrative",
                    model="llama-3.1-8b-instant",
                    prompt=prompt,
                    response=narrative,
                    tokens=usage
                )
            return narrative
        except Exception as e:
            return f"Failed to generate LLM narrative: {e}"
