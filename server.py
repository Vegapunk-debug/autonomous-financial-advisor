#!/usr/bin/env python3
"""FastAPI server for Financial Advisor Agent dashboard."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from src.agent import FinancialAdvisorAgent

app = FastAPI(title="AlphaReason AI")

BASE_DIR = Path(__file__).parent
agent = FinancialAdvisorAgent(str(BASE_DIR / "data"))


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/portfolio/{portfolio_id}")
async def get_portfolio_analysis(portfolio_id: str, model: str = "llama-3.1-8b-instant"):
    try:
        result = agent.run(portfolio_id, model=model)
        return result
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/portfolios")
async def list_portfolios():
    portfolios = agent.loader.get_portfolios()
    return {
        pid: {
            "user_name": p.get("user_name", ""),
            "portfolio_type": p.get("portfolio_type", ""),
            "current_value": p.get("current_value", 0),
            "total_investment": p.get("total_investment", 0),
            "risk_profile": p.get("risk_profile", ""),
            "description": p.get("description", ""),
        }
        for pid, p in portfolios.items()
    }


@app.get("/api/market")
async def get_market_data():
    market = agent.loader.get_market_data()
    historical = agent.loader.get_historical_data()
    return {
        "indices": market.get("indices", {}),
        "sector_performance": market.get("sector_performance", {}),
        "market_breadth": historical.get("market_breadth", {}),
        "fii_dii": historical.get("fii_dii_data", {}),
        "index_history": historical.get("index_history", {}),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
