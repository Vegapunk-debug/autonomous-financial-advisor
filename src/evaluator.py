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