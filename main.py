#!/usr/bin/env python3
"""
Autonomous Financial Advisor Agent
===================================
Analyzes market data, news, and user portfolios to generate
causal explanations of portfolio performance.

Usage:
    python main.py                          # Analyze all portfolios
    python main.py --portfolio PORTFOLIO_001 # Analyze specific portfolio
    python main.py --json                    # Output raw JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import FinancialAdvisorAgent


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaReason AI: Autonomous Financial Advisor Agent")
    parser.add_argument(
        "--portfolio", "-p",
        help="Portfolio ID to analyze (e.g., PORTFOLIO_001). Omit for all.",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./data",
        help="Path to data directory (default: ./data)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.verbose)

    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Try relative to script location
        data_dir = Path(__file__).parent / "data"
    if not data_dir.exists():
        print(f"Error: Data directory not found at {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    agent = FinancialAdvisorAgent(str(data_dir))

    portfolio_ids = (
        [args.portfolio] if args.portfolio
        else ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"]
    )

    for pid in portfolio_ids:
        try:
            result = agent.run(pid)

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(agent.format_briefing_text(result))
                if pid != portfolio_ids[-1]:
                    print("\n")

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
        except Exception as e:
            logging.getLogger("financial_advisor").exception(f"Failed to analyze {pid}")
            print(f"Error analyzing {pid}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
