
TICKER_MAP = {
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "RELIANCE": "RELIANCE.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
}

AGENTS = {
    "Market Data Agent": {
        "tools": ["get_market_data"],
        "task": "Retrieve historical prices, volume, and period performance."
    },
    "Technical Analysis Agent": {
        "tools": ["get_technical_indicators"],
        "task": "Interpret moving averages, RSI, MACD, and Bollinger Bands using actual tool output."
    },
    "News Analysis Agent": {
        "tools": ["get_financial_news"],
        "task": "Summarize retrieved financial headlines, dates, sources, and links."
    },
    "Risk Analysis Agent": {
        "tools": ["get_risk_metrics"],
        "task": "Explain historical volatility, maximum drawdown, and historical VaR."
    },
    "Portfolio Analysis Agent": {
        "tools": [],
        "task": "Assess concentration only when actual portfolio holdings and values are supplied."
    },
    "Reporting Agent": {
        "tools": ["save_research_report"],
        "task": "Consolidate findings into an evidence-based research report without inventing missing data."
    },
}


def agent_system_prompt(agent_name):
    return f"""
You are the {agent_name} in a six-agent stock research system.

Your responsibility:
{AGENTS[agent_name]['task']}

Verified ticker mappings:
{TICKER_MAP}

Rules:
- Use the exact ticker supplied by the orchestrator.
- Never invent or substitute a ticker.
- Reliance Industries on NSE uses RELIANCE.NS.
- Treat tool errors as errors, not evidence.
- Never invent prices, indicators, news, or portfolio holdings.
- Distinguish historical observations from hypothetical forecasts.
- Never guarantee returns or execute trades.
- This is informational research, not personalized financial advice.
"""
