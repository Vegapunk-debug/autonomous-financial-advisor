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

        analytics = self._portfolio.get("analytics", {})
        day_summary = analytics.get("day_summary", {})

        gainers = [h for h in holdings if h.day_change_pct > 0]
        losers = [h for h in holdings if h.day_change_pct < 0]
        top_gainer = max(gainers, key=lambda h: h.day_change_pct) if gainers else None
        top_loser = min(losers, key=lambda h: h.day_change_pct) if losers else None

        return PortfolioSummary(
            portfolio_id=self._portfolio.get("user_id", ""),
            user_name=self._portfolio.get("user_name", ""),
            portfolio_type=self._portfolio.get("portfolio_type", ""),
            total_value=self._portfolio.get("current_value", 0),
            day_pnl_abs=day_summary.get("day_change_absolute", 0),
            day_pnl_pct=day_summary.get("day_change_percent", 0),
            sector_allocation=sector_alloc,
            asset_allocation=asset_alloc,
            holdings=holdings,
            risk_alerts=risk_alerts,
            top_gainer=top_gainer,
            top_loser=top_loser,
        )

    def _analyze_holdings(self) -> list[HoldingAnalysis]:
        results: list[HoldingAnalysis] = []
        total_value = self._portfolio.get("current_value", 1)
        holdings = self._portfolio.get("holdings", {})

        for stock in holdings.get("stocks", []):
            weight = stock.get("weight_in_portfolio", 0)
            day_pct = stock.get("day_change_percent", 0)
            results.append(HoldingAnalysis(
                symbol=stock["symbol"],
                name=stock.get("name", stock["symbol"]),
                asset_type="STOCK",
                sector=stock.get("sector", "OTHER"),
                weight=weight,
                day_change_pct=day_pct,
                day_change_abs=stock.get("day_change", 0),
                total_gain_pct=stock.get("gain_loss_percent", 0),
                contribution_to_day_pnl=weight * day_pct / 100,
            ))

        for mf in holdings.get("mutual_funds", []):
            weight = mf.get("weight_in_portfolio", 0)
            day_pct = mf.get("day_change_percent", 0)
            mf_detail = self._mf.get(mf.get("scheme_code", ""), {})
            category = mf.get("category", mf_detail.get("category", "DIVERSIFIED"))
            results.append(HoldingAnalysis(
                symbol=mf.get("scheme_code", ""),
                name=mf.get("scheme_name", ""),
                asset_type="MUTUAL_FUND",
                sector=f"MF_{category}",
                weight=weight,
                day_change_pct=day_pct,
                day_change_abs=mf.get("day_change", 0),
                total_gain_pct=mf.get("gain_loss_percent", 0),
                contribution_to_day_pnl=weight * day_pct / 100,
            ))

        results.sort(key=lambda h: abs(h.contribution_to_day_pnl), reverse=True)
        return results

    def _compute_sector_allocation(self, holdings: list[HoldingAnalysis]) -> dict[str, float]:
        alloc: dict[str, float] = {}
        for h in holdings:
            sector = h.sector if h.asset_type == "STOCK" else "MUTUAL_FUNDS"
            alloc[sector] = alloc.get(sector, 0) + h.weight
        return dict(sorted(alloc.items(), key=lambda x: x[1], reverse=True))

    def _compute_asset_allocation(self) -> dict[str, float]:
        analytics = self._portfolio.get("analytics", {})
        return analytics.get("asset_type_allocation", {})

    def _detect_risks(self, holdings: list[HoldingAnalysis],
                      sector_alloc: dict[str, float]) -> list[RiskAlert]:
        alerts: list[RiskAlert] = []

        # Sector concentration
        for sector, weight in sector_alloc.items():
            if sector == "MUTUAL_FUNDS":
                continue
            if weight >= self.CONCENTRATION_THRESHOLD:
                alerts.append(RiskAlert(
                    level="CRITICAL",
                    category="CONCENTRATION",
                    message=f"Sector concentration: {sector} at {weight:.1f}% of portfolio",
                    details={"sector": sector, "weight": weight},
                ))
            elif weight >= self.CONCENTRATION_THRESHOLD * 0.75:
                alerts.append(RiskAlert(
                    level="WARNING",
                    category="CONCENTRATION",
                    message=f"Elevated sector exposure: {sector} at {weight:.1f}%",
                    details={"sector": sector, "weight": weight},
                ))

        # Single stock concentration
        for h in holdings:
            if h.asset_type == "STOCK" and h.weight >= self.SINGLE_STOCK_THRESHOLD:
                alerts.append(RiskAlert(
                    level="CRITICAL",
                    category="SINGLE_STOCK",
                    message=f"Single stock risk: {h.symbol} at {h.weight:.1f}% of portfolio",
                    details={"symbol": h.symbol, "weight": h.weight},
                ))

        # Beta risk
        risk_metrics = self._portfolio.get("analytics", {}).get("risk_metrics", {})
        beta = risk_metrics.get("beta", 1.0)
        if beta >= self.HIGH_BETA_THRESHOLD:
            alerts.append(RiskAlert(
                level="WARNING",
                category="VOLATILITY",
                message=f"Portfolio beta {beta:.2f} indicates high market sensitivity",
                details={"beta": beta},
            ))

        # Rate sensitivity check
        rate_sensitive_weight = sum(
            h.weight for h in holdings
            if h.asset_type == "STOCK" and h.sector in (
                "BANKING", "REALTY", "FINANCIAL_SERVICES", "AUTOMOBILE", "INFRASTRUCTURE"
            )
        )
        if rate_sensitive_weight >= 50:
            alerts.append(RiskAlert(
                level="WARNING",
                category="RATE_SENSITIVITY",
                message=f"Rate-sensitive sectors at {rate_sensitive_weight:.1f}% — vulnerable to RBI policy changes",
                details={"weight": rate_sensitive_weight},
            ))

        alerts.sort(key=lambda a: {"CRITICAL": 3, "WARNING": 2, "INFO": 1}.get(a.level, 0), reverse=True)
        return alerts
