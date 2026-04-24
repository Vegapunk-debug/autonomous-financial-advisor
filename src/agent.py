"""Main agent orchestrator — ties all layers together with tracing."""
from __future__ import annotations

import logging
from typing import Any

from .data_loader import DataLoader
from .market_intelligence import MarketIntelligence
from .portfolio_analytics import PortfolioAnalytics
from .reasoning_engine import ReasoningEngine, AdvisorBriefing
from .evaluator import BriefingEvaluator, EvalReport
from .observability import Tracer

logger = logging.getLogger("financial_advisor")


class FinancialAdvisorAgent:
    def __init__(self, data_dir: str = "./data") -> None:
        self.loader = DataLoader(data_dir)
        self.tracer = Tracer()
        self.evaluator = BriefingEvaluator()

    def run(self, portfolio_id: str, model: str = "llama-3.1-8b-instant") -> dict[str, Any]:
        """Execute full advisory pipeline for a portfolio. Returns structured result."""
        self.tracer.start_trace(
            f"advisory_{portfolio_id}",
            metadata={"portfolio_id": portfolio_id},
        )

        try:
            # Phase 1: Load data
            with self.tracer.span("data_loading") as span:
                market_data = self.loader.get_market_data()
                news_data = self.loader.get_news()
                historical = self.loader.get_historical_data()
                sector_mapping = self.loader.get_sector_mapping()
                portfolio = self.loader.get_portfolio(portfolio_id)
                mutual_funds = self.loader.get_mutual_funds()

                if not portfolio:
                    raise ValueError(f"Portfolio {portfolio_id} not found")

                span.output_data = {
                    "stocks_loaded": len(market_data.get("stocks", {})),
                    "news_loaded": len(news_data),
                    "portfolio": portfolio.get("user_name", ""),
                }

            # Phase 2: Market Intelligence
            with self.tracer.span("market_intelligence") as span:
                intel = MarketIntelligence(market_data, news_data, historical, sector_mapping)
                market_summary = intel.build_summary()
                span.output_data = {
                    "sentiment": market_summary["market_sentiment"]["overall"],
                    "high_impact_news": len(market_summary["high_impact_news"]),
                    "conflicts": len(market_summary["conflicting_signals"]),
                }

            # Phase 3: Portfolio Analytics
            with self.tracer.span("portfolio_analytics") as span:
                pa = PortfolioAnalytics(portfolio, market_data, mutual_funds)
                portfolio_summary = pa.analyze()
                span.output_data = {
                    "day_pnl": portfolio_summary.day_pnl_pct,
                    "risk_alerts": len(portfolio_summary.risk_alerts),
                    "holdings_count": len(portfolio_summary.holdings),
                }

            # Phase 4: Reasoning
            with self.tracer.span("reasoning") as span:
                engine = ReasoningEngine(intel, pa, self.tracer)
                briefing = engine.generate_briefing()
                narrative = engine.generate_narrative(model=model)
                span.output_data = {
                    "causal_chains": len(briefing.causal_chains),
                    "conflicts_resolved": len(briefing.conflicts),
                    "confidence": briefing.confidence_score,
                }

            # Phase 5: Self-Evaluation
            with self.tracer.span("evaluation") as span:
                news_for_eval = {"news": news_data}
                eval_report = self.evaluator.evaluate(briefing, portfolio, news_for_eval, narrative=narrative)
                span.output_data = {
                    "overall_score": eval_report.overall_score,
                    "grade": eval_report.grade,
                }

            result = self._format_output(briefing, eval_report, narrative)
            return result

        finally:
            self.tracer.flush()

    def _format_output(self, briefing: AdvisorBriefing, eval_report: EvalReport, narrative: str = "") -> dict[str, Any]:
        return {
            "briefing": {
                "date": briefing.date,
                "user": briefing.user_name,
                "portfolio_type": briefing.portfolio_type,
                "market_overview": briefing.market_overview,
                "portfolio_pnl": briefing.portfolio_pnl,
                "causal_chains": [
                    {
                        "trigger": c.trigger,
                        "chain": c.chain,
                        "impact": c.portfolio_impact,
                        "confidence": c.confidence,
                        "affected": c.affected_holdings,
                    }
                    for c in briefing.causal_chains
                ],
                "conflicts": [
                    {
                        "symbol": c.symbol,
                        "positive": c.positive_signals,
                        "negative": c.negative_signals,
                        "resolution": c.resolution,
                        "assessment": c.net_assessment,
                    }
                    for c in briefing.conflicts
                ],
                "risk_alerts": briefing.risk_alerts,
                "key_takeaway": briefing.key_takeaway,
                "confidence_score": briefing.confidence_score,
                "narrative": narrative,
            },
            "evaluation": {
                "overall_score": eval_report.overall_score,
                "grade": eval_report.grade,
                "dimensions": [
                    {"name": s.dimension, "score": s.score, "reason": s.reason}
                    for s in eval_report.scores
                ],
                "suggestions": eval_report.suggestions,
            },
            "trace": self.tracer.get_summary(),
        }

    def format_briefing_text(self, result: dict) -> str:
        """Format result as human-readable briefing text."""
        b = result["briefing"]
        e = result["evaluation"]
        t = result.get("trace", {})

        lines = [
            "=" * 70,
            f"  FINANCIAL ADVISOR BRIEFING — {b['date']}",
            f"  Portfolio: {b['user']} ({b['portfolio_type']})",
            "=" * 70,
            "",
            "--- MARKET OVERVIEW ---",
            b["market_overview"],
            "",
            "--- PORTFOLIO P&L ---",
            b["portfolio_pnl"],
            "",
        ]

        if b["causal_chains"]:
            lines.append("--- CAUSAL ANALYSIS ---")
            for i, chain in enumerate(b["causal_chains"], 1):
                lines.append(f"\n  Chain {i} (confidence: {chain['confidence']:.0%}):")
                for step in chain["chain"]:
                    lines.append(f"    -> {step}")
                lines.append(f"    => {chain['impact']}")
                lines.append(f"    Affected: {', '.join(chain['affected'])}")
            lines.append("")

        if b["conflicts"]:
            lines.append("--- CONFLICTING SIGNALS ---")
            for c in b["conflicts"]:
                lines.append(f"\n  {c['symbol']} [{c['assessment']}]:")
                lines.append(f"    Resolution: {c['resolution']}")
            lines.append("")

        if b["risk_alerts"]:
            lines.append("--- RISK ALERTS ---")
            for alert in b["risk_alerts"]:
                lines.append(f"  ⚠ {alert}")
            lines.append("")

        lines.extend([
            "--- KEY TAKEAWAY ---",
            b["key_takeaway"],
            "",
            "--- LLM ADVISORY NARRATIVE ---",
            b.get("narrative", ""),
            "",
            f"--- REASONING QUALITY: {e['grade']} ({e['overall_score']:.0%}) ---",
        ])
        for dim in e["dimensions"]:
            bar = "█" * int(dim["score"] * 10) + "░" * (10 - int(dim["score"] * 10))
            lines.append(f"  {dim['name']:25s} {bar} {dim['score']:.0%}  {dim['reason']}")

        if e["suggestions"]:
            lines.append("\n  Improvement suggestions:")
            for s in e["suggestions"]:
                lines.append(f"    - {s}")

        lines.extend([
            "",
            f"--- TRACE ---",
            f"  Spans: {t.get('total_spans', 0)} | Duration: {t.get('total_duration_ms', 0):.0f}ms",
            f"  Tokens: {t.get('token_usage', {}).get('total', 0)}",
            "=" * 70,
        ])

        return "\n".join(lines)
