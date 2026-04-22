"""Data loader for market data, news, portfolios, and sector mappings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataLoader:
    def __init__(self, data_dir: str = "./data") -> None:
        self._dir = Path(data_dir)
        self._cache: dict[str, Any] = {}

    def _load(self, name: str) -> dict:
        if name not in self._cache:
            with open(self._dir / f"{name}.json") as f:
                self._cache[name] = json.load(f)
        return self._cache[name]

    def get_market_data(self) -> dict:
        return self._load("market_data")

    def get_news(self) -> list[dict]:
        data = self._load("news_data")
        return data.get("news", data.get("articles", []))

    def get_portfolios(self) -> dict:
        return self._load("portfolios").get("portfolios", {})

    def get_portfolio(self, portfolio_id: str) -> dict | None:
        return self.get_portfolios().get(portfolio_id)

    def get_mutual_funds(self) -> dict:
        return self._load("mutual_funds").get("mutual_funds", {})

    def get_historical_data(self) -> dict:
        return self._load("historical_data")

    def get_sector_mapping(self) -> dict:
        return self._load("sector_mapping")

    def get_sector_info(self, sector: str) -> dict | None:
        return self.get_sector_mapping().get("sectors", {}).get(sector)

    def get_stock_with_context(self, symbol: str) -> dict:
        """Return stock data enriched with relevant news and sector info."""
        market = self.get_market_data()
        stock = market.get("stocks", {}).get(symbol, {})
        sector = stock.get("sector", "")
        sector_perf = market.get("sector_performance", {}).get(sector, {})
        sector_info = self.get_sector_info(sector) or {}
        news = [
            n for n in self.get_news()
            if symbol in n.get("entities", {}).get("stocks", [])
            or sector in n.get("entities", {}).get("sectors", [])
        ]
        history = self.get_historical_data().get("stock_history", {}).get(symbol, {})
        return {
            "stock": stock,
            "sector_performance": sector_perf,
            "sector_info": sector_info,
            "related_news": news,
            "history": history,
        }
