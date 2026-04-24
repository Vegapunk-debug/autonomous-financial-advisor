"""Phase 4b: Self-Evaluation — rule-based + LLM scoring of reasoning quality."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalScore:
    dimension: str
    score: float  # 0-1
    reason: str
    weight: float = 1.0


@dataclass
class EvalReport:
    overall_score: float
    grade: str  # A/B/C/D/F
    scores: list[EvalScore]
    suggestions: list[str]


class BriefingEvaluator:
    """Evaluates quality of generated briefings using rule-based checks and optional LLM grading."""

    def evaluate(self, briefing: Any, portfolio_data: dict, market_data: dict) -> EvalReport:
        scores: list[EvalScore] = []

        scores.append(self._eval_causal_depth(briefing))
        scores.append(self._eval_coverage(briefing, portfolio_data))
        scores.append(self._eval_conflict_handling(briefing, market_data))
        scores.append(self._eval_risk_identification(briefing, portfolio_data))
        scores.append(self._eval_confidence_calibration(briefing))
        scores.append(self._eval_conciseness(briefing))

        # Weighted average
        total_weight = sum(s.weight for s in scores)
        overall = sum(s.score * s.weight for s in scores) / max(total_weight, 1)

        grade = self._score_to_grade(overall)
        suggestions = self._generate_suggestions(scores)

        return EvalReport(
            overall_score=round(overall, 3),
            grade=grade,
            scores=scores,
            suggestions=suggestions,
        )

    def _eval_causal_depth(self, briefing: Any) -> EvalScore:
        """Check if causal chains have sufficient depth (trigger -> sector -> stock -> portfolio)."""
        chains = briefing.causal_chains
        if not chains:
            return EvalScore("causal_depth", 0.0, "No causal chains generated", weight=2.0)

        avg_depth = sum(len(c.chain) for c in chains) / len(chains)
        chains_with_holdings = sum(1 for c in chains if c.affected_holdings)
        coverage_ratio = chains_with_holdings / len(chains)

        score = min(1.0, (avg_depth / 4) * 0.6 + coverage_ratio * 0.4)

        reason = (
            f"{len(chains)} chains, avg depth {avg_depth:.1f} steps, "
            f"{chains_with_holdings}/{len(chains)} linked to holdings"
        )
        return EvalScore("causal_depth", round(score, 2), reason, weight=2.0)

    def _eval_coverage(self, briefing: Any, portfolio: dict) -> EvalScore:
        """Check how many portfolio holdings are covered by the analysis."""
        holdings = portfolio.get("holdings", {})
        stock_symbols = {s["symbol"] for s in holdings.get("stocks", [])}

        explained = set()
        for chain in briefing.causal_chains:
            explained.update(chain.affected_holdings)
        for conflict in briefing.conflicts:
            explained.add(conflict.symbol)

        if not stock_symbols:
            return EvalScore("coverage", 1.0, "No stocks to cover", weight=1.5)

        ratio = len(explained & stock_symbols) / len(stock_symbols)
        uncovered = stock_symbols - explained
        reason = f"Covered {len(explained & stock_symbols)}/{len(stock_symbols)} stocks"
        if uncovered:
            reason += f". Uncovered: {', '.join(list(uncovered)[:3])}"

        return EvalScore("coverage", round(ratio, 2), reason, weight=1.5)

    def _eval_conflict_handling(self, briefing: Any, market: dict) -> EvalScore:
        """Check if conflicting signals are properly identified and resolved."""
        news = market.get("news", []) if "news" in market else []
        actual_conflicts = sum(1 for n in news if n.get("conflict_flag"))

        detected = len(briefing.conflicts)
        if actual_conflicts == 0:
            return EvalScore("conflict_handling", 1.0, "No conflicts to detect", weight=1.0)

        detection_rate = min(detected / max(actual_conflicts, 1), 1.0)
        has_resolution = all(c.resolution for c in briefing.conflicts)
        resolution_bonus = 0.2 if has_resolution and detected > 0 else 0

        score = min(1.0, detection_rate * 0.8 + resolution_bonus)
        reason = f"Detected {detected}/{actual_conflicts} conflicts"
        if has_resolution and detected > 0:
            reason += ", all with resolutions"

        return EvalScore("conflict_handling", round(score, 2), reason, weight=1.0)

    def _eval_risk_identification(self, briefing: Any, portfolio: dict) -> EvalScore:
        """Check if critical risks are surfaced."""
        analytics = portfolio.get("analytics", {})
        risk_metrics = analytics.get("risk_metrics", {})
        has_concentration = risk_metrics.get("concentration_risk", False)

        alerts = briefing.risk_alerts
        if not has_concentration and not alerts:
            return EvalScore("risk_identification", 0.8, "No critical risks, none flagged", weight=1.0)

        if has_concentration:
            found_concentration = any("concentration" in a.lower() or "sector" in a.lower() for a in alerts)
            if found_concentration:
                return EvalScore("risk_identification", 1.0,
                                "Concentration risk correctly identified", weight=1.0)
            return EvalScore("risk_identification", 0.2,
                            "Missed concentration risk", weight=1.0)

        return EvalScore("risk_identification", 0.7,
                        f"{len(alerts)} risk(s) flagged", weight=1.0)

    def _eval_confidence_calibration(self, briefing: Any) -> EvalScore:
        """Check if confidence scores are reasonable (not all 1.0 or all 0.0)."""
        if not briefing.causal_chains:
            return EvalScore("confidence_calibration", 0.5, "No chains to calibrate", weight=0.5)

        confidences = [c.confidence for c in briefing.causal_chains]
        avg = sum(confidences) / len(confidences)
        spread = max(confidences) - min(confidences) if len(confidences) > 1 else 0

        # Good calibration: spread > 0, avg not extreme
        score = 0.5
        if 0.3 <= avg <= 0.85:
            score += 0.25
        if spread >= 0.05:
            score += 0.25

        reason = f"Avg confidence: {avg:.2f}, spread: {spread:.2f}"
        return EvalScore("confidence_calibration", round(score, 2), reason, weight=0.5)

    def _eval_conciseness(self, briefing: Any) -> EvalScore:
        """Penalize overly verbose or overly sparse briefings."""
        text_len = len(briefing.market_overview) + len(briefing.portfolio_pnl) + len(briefing.key_takeaway)

        if text_len < 100:
            return EvalScore("conciseness", 0.3, "Briefing too sparse", weight=0.5)
        if text_len > 3000:
            return EvalScore("conciseness", 0.5, "Briefing may be too verbose", weight=0.5)
        return EvalScore("conciseness", 1.0, f"Briefing length appropriate ({text_len} chars)", weight=0.5)

    def _score_to_grade(self, score: float) -> str:
        if score >= 0.85:
            return "A"
        if score >= 0.7:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.4:
            return "D"
        return "F"

    def _generate_suggestions(self, scores: list[EvalScore]) -> list[str]:
        suggestions = []
        for s in scores:
            if s.score < 0.5:
                suggestions.append(f"Improve {s.dimension}: {s.reason}")
        return suggestions
