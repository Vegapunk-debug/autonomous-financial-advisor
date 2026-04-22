Backend Engineering Challenge: Autonomous Financial Advisor Agent
1. Overview
The objective is to build an Autonomous Financial Advisor Agent that doesn't just report data but reasons through it. The agent will ingest market data and news, analyze a user’s portfolio, and generate a concise, causal explanation of how external events impacted their specific holdings.
The Core Challenge
Moving beyond simple API integration to build a Reasoning & Causality Layer that can link:
Macro News → Sector Trends → Individual Stock Performance → User Portfolio Impact.

2. Technical Requirements
Phase 1: Market Intelligence Layer
Trend Analysis: Analyze index movements (NIFTY 50, SENSEX) to determine market sentiment (Bullish/Bearish/Neutral).
Sector Extraction: Derive sector-level trends dynamically from stock data.
News Processing: Classify news by Sentiment and Scope (Market-wide, Sector-specific, or Stock-specific).
Phase 2: Portfolio Analytics Engine
Given a mock portfolio (Stocks + Mutual Funds), compute:
Daily P&L: Absolute and percentage changes.
Asset Allocation: Breakdown by sector and asset type.
Risk Detection: Identify concentration risks (e.g., >40% exposure to a single sector).
Phase 3: Autonomous Reasoning (The "Agent" Logic)
The agent must decide what is relevant. Instead of a data dump, it should:
Causal Linking: Explain why a portfolio moved (e.g., "Your portfolio fell 1.2% primarily because the Banking sector reacted negatively to the RBI interest rate news, affecting your 30% holding in HDFC Bank.")
Conflict Resolution: Handle edge cases like positive news paired with falling prices (explain the ambiguity).
Prioritization: Highlight only high-impact signals.
Phase 4: Observability & Evaluation
Tracing: Integrate Langfuse (or equivalent) to track prompts, responses, and token usage.
Self-Evaluation: Implement a basic evaluation step (LLM-based or rule-based) to score the "Reasoning Quality" of the generated briefing.

3. Architecture & Performance
Modularity: Separate the Market Ingestion, Portfolio Analytics, and Reasoning layers.
Latency: The end-to-end response should be optimized (avoid redundant LLM calls).
Reliability: Use type hints, handle missing data gracefully, and provide a Confidence Score for explanations.

4. Deliverables
Candidates are expected to submit:
GitHub Repository: Containing clean, documented, and modular code.
README: Instructions on how to run the agent (CLI or simple script loop).
Demo video & Link to try: A 2–3 minute walkthrough of the code architecture and a live run of the agent reacting to different portfolio samples.

5. Evaluation Rubric
Criteria
Weight
What we look for
Reasoning Quality
35%
Depth of causal links (News → Sector → Stock).
Code Design
20%
Modularity, extensibility, and use of type hints.
Observability
15%
Integration with Langfuse/tracing and structured logging.
Edge Case Handling
15%
Handling conflicting signals or skewed portfolios.
Evaluation Layer
15%
How the agent "grades" its own output for accuracy.


6. Provided Mock Data Inputs
Market Data: Daily % change for Indices and Stock-to-Sector mapping.
News Data: Headlines and summaries with entity tags.
User Portfolios: 3 samples (1 Diversified, 1 Sector-heavy, 1 Conservative MF-heavy).

Submission Instructions
Please share your GitHub link and Demo Video & Link to try within 48 hours. 
