"""Phase 2: Portfolio Analytics Engine — P&L, allocation, risk detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HoldingAnalysis:
    symbol: str
    name: str
    asset_type: str  # STOCK or MUTUAL_FUND
    sector: str
    weight: float
    day_change_pct: float
    day_change_abs: float
    total_gain_pct: float
    contribution_to_day_pnl: float  # weighted contribution


@dataclass
class RiskAlert:
    level: str  # CRITICAL / WARNING / INFO
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSummary:
    portfolio_id: str
    user_name: str
    portfolio_type: str
    total_value: float
    day_pnl_abs: float
    day_pnl_pct: float
    sector_allocation: dict[str, float]
    asset_allocation: dict[str, float]
    holdings: list[HoldingAnalysis]
    risk_alerts: list[RiskAlert]
    top_gainer: HoldingAnalysis | None
    top_loser: HoldingAnalysis | None


class PortfolioAnalytics:
    CONCENTRATION_THRESHOLD = 40.0  # sector concentration warning
    SINGLE_STOCK_THRESHOLD = 20.0   # single stock warning
    HIGH_BETA_THRESHOLD = 1.3

    def __init__(self, portfolio: dict, market_data: dict, mutual_funds: dict) -> None:
        self._portfolio = portfolio
        self._market = market_data
        self._mf = mutual_funds

    def analyze(self) -> PortfolioSummary:
        holdings = self._analyze_holdings()
        sector_alloc = self._compute_sector_allocation(holdings)
        asset_alloc = self._compute_asset_allocation()
        risk_alerts = self._detect_risks(holdings, sector_alloc)
