
import json
import math
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("StockMarketMAS")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "repository"
REPORTS.mkdir(exist_ok=True)


TICKER_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "GOOGLE": "GOOGL",
}


def normalize_ticker(ticker: str) -> str:
    """Resolve supported company names to their Yahoo Finance tickers."""
    value = ticker.strip().upper()

    if value in TICKER_MAP:
        return TICKER_MAP[value]

    if value in set(TICKER_MAP.values()):
        return value

    raise ValueError(
        f"Unsupported ticker or company: {ticker}. "
        "Use a supported company name or ticker."
    )


def history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download historical prices for a verified ticker."""
    ticker = normalize_ticker(ticker)

    allowed_periods = {
        "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"
    }

    if period not in allowed_periods:
        raise ValueError(
            f"Unsupported period: {period}. "
            f"Choose from {sorted(allowed_periods)}."
        )

    df = yf.Ticker(ticker).history(
        period=period,
        auto_adjust=True
    )

    if df.empty:
        raise ValueError(f"No market data found for {ticker}.")

    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"No valid closing prices found for {ticker}.")

    return df


def safe(value):
    """Convert a numeric value to a JSON-safe rounded number."""
    if pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), 4)


@mcp.tool()
def get_market_data(ticker: str, period: str = "1y") -> str:
    """Retrieve historical OHLC prices, volume, and period performance."""
    try:
        ticker = normalize_ticker(ticker)
        df = history(ticker, period)

        first = df.iloc[0]
        last = df.iloc[-1]

        change = (last["Close"] / first["Close"] - 1) * 100

        return json.dumps({
            "ticker": ticker,
            "period": period,
            "data_points": len(df),
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "start_close": safe(first["Close"]),
            "latest_close": safe(last["Close"]),
            "period_change_percent": safe(change),
            "latest_volume": int(last["Volume"]),
            "recent_sessions": [
                {
                    "date": str(index.date()),
                    "close": safe(row["Close"]),
                    "volume": int(row["Volume"]),
                    "high": safe(row["High"]),
                    "low": safe(row["Low"])
                }
                for index, row in df.tail(10).iterrows()
            ],
            "source": "Yahoo Finance via yfinance",
            "retrieved_at": datetime.now().isoformat()
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_technical_indicators(ticker: str, period: str = "1y") -> str:
    """Calculate SMA20/50, EMA12/26, RSI14, MACD and Bollinger Bands."""
    try:
        ticker = normalize_ticker(ticker)
        df = history(ticker, period)
        close = df["Close"]

        df["SMA20"] = close.rolling(20).mean()
        df["SMA50"] = close.rolling(50).mean()
        df["EMA12"] = close.ewm(span=12, adjust=False).mean()
        df["EMA26"] = close.ewm(span=26, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / 14, min_periods=14, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = 100 - 100 / (1 + rs)

        # Handle periods where average losses are zero.
        df.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
        df.loc[(avg_loss == 0) & (avg_gain == 0), "RSI14"] = 50

        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_signal"] = df["MACD"].ewm(
            span=9, adjust=False
        ).mean()

        df["BB_mid"] = close.rolling(20).mean()
        std = close.rolling(20).std()
        df["BB_upper"] = df["BB_mid"] + 2 * std
        df["BB_lower"] = df["BB_mid"] - 2 * std

        last = df.iloc[-1]

        return json.dumps({
            "ticker": ticker,
            "latest_close": safe(last["Close"]),
            "SMA20": safe(last["SMA20"]),
            "SMA50": safe(last["SMA50"]),
            "EMA12": safe(last["EMA12"]),
            "EMA26": safe(last["EMA26"]),
            "RSI14": safe(last["RSI14"]),
            "MACD": safe(last["MACD"]),
            "MACD_signal": safe(last["MACD_signal"]),
            "Bollinger_upper": safe(last["BB_upper"]),
            "Bollinger_middle": safe(last["BB_mid"]),
            "Bollinger_lower": safe(last["BB_lower"]),
            "note": "Indicators are historical signals, not guarantees."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_risk_metrics(ticker: str, period: str = "1y") -> str:
    """Estimate historical volatility, maximum drawdown and daily historical VaR."""
    try:
        ticker = normalize_ticker(ticker)
        df = history(ticker, period)
        returns = df["Close"].pct_change().dropna()

        if len(returns) < 2:
            raise ValueError("Insufficient data for risk calculations.")

        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * math.sqrt(252)

        return json.dumps({
            "ticker": ticker,
            "period": period,
            "daily_volatility_percent": safe(daily_volatility * 100),
            "annualized_volatility_percent": safe(
                annualized_volatility * 100
            ),
            "maximum_drawdown_percent": safe(drawdown.min() * 100),
            "historical_daily_VaR_95_percent": safe(
                np.percentile(returns, 5) * 100
            ),
            "historical_daily_VaR_99_percent": safe(
                np.percentile(returns, 1) * 100
            ),
            "observations": len(returns),
            "method": (
                "Historical daily returns; annualization uses "
                "252 trading days."
            ),
            "warning": (
                "Historical risk estimates do not guarantee "
                "future outcomes."
            )
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@mcp.tool()
def get_financial_news(company: str, limit: int = 8) -> str:
    """Fetch financial news headlines and source links through Google News RSS."""
    try:
        if not company.strip():
            raise ValueError("Company name or ticker is required.")

        limit = max(1, min(int(limit), 15))

        # Resolve known company names for consistent news searches.
        normalized_company = company.strip().upper()
        search_company = TICKER_MAP.get(
            normalized_company,
            company.strip()
        )

        # Use readable company names for Indian stock tickers.
        readable_names = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "TATAMOTORS.NS": "Tata Motors",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "NVDA": "NVIDIA",
            "GOOGL": "Google",
        }

        search_company = readable_names.get(
            search_company,
            readable_names.get(normalized_company, search_company)
        )

        url = (
            "https://news.google.com/rss/search?q="
            + quote(search_company + " stock market finance")
            + "&hl=en-IN&gl=IN&ceid=IN:en"
        )

        feed = feedparser.parse(url)

        articles = [
            {
                "headline": item.get("title", ""),
                "published": item.get("published", "Unknown"),
                "source": item.get("source", {}).get("title", "Unknown"),
                "link": item.get("link", "")
            }
            for item in feed.entries[:limit]
        ]

        return json.dumps({
            "company": search_company,
            "articles": articles,
            "count": len(articles),
            "source": "Google News RSS",
            "note": "Headlines may be incomplete; verify important claims."
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "company": company})


@mcp.tool()
def save_research_report(ticker: str, report: str) -> str:
    """Save a generated research report as a text file in repository/."""
    try:
        ticker = normalize_ticker(ticker)

        name = (
            f"{ticker}_{datetime.now():%Y%m%d_%H%M%S}_report.txt"
        )
        path = REPORTS / name

        path.write_text(
            "STOCK MARKET RESEARCH REPORT\n"
            f"Ticker: {ticker}\n"
            f"Generated: {datetime.now().isoformat()}\n\n"
            "DISCLAIMER: Informational analysis only, not personalized "
            "financial advice. Historical performance does not guarantee "
            "future results.\n\n"
            + report,
            encoding="utf-8"
        )

        return json.dumps({
            "status": "saved",
            "file": str(path)
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
