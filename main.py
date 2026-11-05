
import asyncio
import json
import re
import sys
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.specialized_agents import (
    AGENTS,
    TICKER_MAP,
    agent_system_prompt,
)

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_servers" / "stock_market_server.py"
MODEL = "llama3.2:latest"

MARKET_TOOLS = {
    "get_market_data",
    "get_technical_indicators",
    "get_risk_metrics",
}


def extract_ticker(request):
    text = request.upper()

    for company in sorted(TICKER_MAP, key=len, reverse=True):
        if re.search(
            r"(?<![A-Z0-9])" + re.escape(company) + r"(?![A-Z0-9])",
            text
        ):
            return TICKER_MAP[company]

    for ticker in sorted(set(TICKER_MAP.values()), key=len, reverse=True):
        if re.search(
            r"(?<![A-Z0-9])" + re.escape(ticker) + r"(?![A-Z0-9])",
            text
        ):
            return ticker

    return None


def company_name(ticker):
    return {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
        "TATAMOTORS.NS": "Tata Motors",
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
        "GOOGL": "Google",
    }.get(ticker, ticker)


def field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def tool_text(result):
    content = field(result, "content", []) or []
    parts = [
        field(item, "text", "")
        for item in content
        if field(item, "text", "")
    ]
    return "\n".join(parts) or str(result)


def ollama_tool(tool):
    return {
        "type": "function",
        "function": {
            "name": field(tool, "name"),
            "description": field(tool, "description", "") or "",
            "parameters": field(tool, "inputSchema", {}) or {},
        },
    }


async def run_agent(
    session,
    agent_name,
    request,
    ticker,
    company,
    findings,
    server_tools,
):
    allowed = AGENTS[agent_name]["tools"]

    selected_tools = [
        tool for tool in server_tools
        if field(tool, "name") in allowed
    ]

    messages = [
        {
            "role": "system",
            "content": (
                agent_system_prompt(agent_name)
                + f"\nVerified ticker: {ticker}"
                + f"\nCompany: {company}"
                + "\nUse only your assigned tools."
                + "\nNever replace the verified ticker."
                + "\nDo not invent tool results."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Request: {request}\n"
                f"Verified ticker: {ticker}\n"
                f"Company: {company}\n"
                f"Previous findings:\n"
                f"{json.dumps(findings, ensure_ascii=False, default=str)}"
            ),
        },
    ]

    actual_results = []

    for _ in range(5):
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=[ollama_tool(t) for t in selected_tools] or None,
            options={"temperature": 0},
        )

        message = field(response, "message")
        content = field(message, "content", "") or ""
        calls = field(message, "tool_calls", []) or []

        assistant_message = {"role": "assistant", "content": content}

        if calls:
            assistant_message["tool_calls"] = calls

        messages.append(assistant_message)

        if not calls:
            return (
                content
                + ("\n\nTool results:\n" + "\n\n".join(actual_results)
                   if actual_results else "")
            ) or "Agent returned no response."

        for call in calls:
            function = field(call, "function", {})
            name = field(function, "name", "")
            arguments = field(function, "arguments", {}) or {}

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            if name not in allowed:
                output = f"Blocked unauthorized tool call: {name}"
            else:
                if name in MARKET_TOOLS:
                    arguments = dict(arguments)
                    arguments["ticker"] = ticker
                    arguments.setdefault("period", "1y")

                elif name == "get_financial_news":
                    arguments = dict(arguments)
                    arguments["company"] = company
                    arguments.setdefault("limit", 8)

                try:
                    result = await session.call_tool(
                        name,
                        arguments=arguments,
                    )
                    output = tool_text(result)

                except Exception as exc:
                    output = f"Tool error: {type(exc).__name__}: {exc}"

            actual_results.append(f"{name}:\n{output}")

            messages.append({
                "role": "tool",
                "name": name or "unknown_tool",
                "content": output,
            })

    return (
        "Agent reached its tool-call limit.\n"
        + "\n\n".join(actual_results)
    )


async def main():
    print("=" * 55)
    print("STOCK MARKET MULTI-AGENT SYSTEM")
    print("=" * 55)

    try:
        available = field(ollama.list(), "models", []) or []

        names = [
            field(model, "name", "") or field(model, "model", "")
            for model in available
        ]

        if not any(
            name.split(":")[0] == "llama3.2"
            for name in names
        ):
            print("Model llama3.2 is missing.")
            print("Run: ollama pull llama3.2")
            return

        print(f"Ollama model {MODEL} is ready.")

    except Exception as exc:
        print(f"Cannot connect to Ollama: {exc}")
        print("Start the Ollama application and try again.")
        return

    if not SERVER.exists():
        print(f"MCP server not found: {SERVER}")
        return

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        cwd=str(ROOT),
    )

    try:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                response = await session.list_tools()
                server_tools = field(response, "tools", []) or []

                print("\nMCP connected. Tools:")

                for tool in server_tools:
                    print(" -", field(tool, "name"))

                print("\nExamples:")
                print("  Analyze Reliance Industries")
                print("  Analyze TCS.NS for six months")
                print("  Analyze NVIDIA")
                print("Type 'exit' to quit.\n")

                while True:
                    try:
                        request = input("Stock research question: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nExiting.")
                        break

                    if request.lower() in {"exit", "quit"}:
                        break

                    if not request:
                        continue

                    ticker = extract_ticker(request)

                    if not ticker:
                        print(
                            "Please specify a supported company or ticker, "
                            "such as Reliance Industries, TCS, Infosys, "
                            "Apple, Microsoft, NVIDIA, or RELIANCE.NS.\n"
                        )
                        continue

                    company = company_name(ticker)

                    print(f"\nCompany: {company}")
                    print(f"Verified ticker: {ticker}")

                    findings = {}

                    for agent_name in [
                        "Market Data Agent",
                        "Technical Analysis Agent",
                        "News Analysis Agent",
                        "Risk Analysis Agent",
                        "Portfolio Analysis Agent",
                    ]:
                        print(f"\n--- {agent_name} ---")

                        result = await run_agent(
                            session=session,
                            agent_name=agent_name,
                            request=request,
                            ticker=ticker,
                            company=company,
                            findings=findings,
                            server_tools=server_tools,
                        )

                        findings[agent_name] = result
                        print(result)

                    print("\n--- Reporting Agent ---")

                    report_messages = [
                        {
                            "role": "system",
                            "content": (
                                agent_system_prompt("Reporting Agent")
                                + f"\nThe verified ticker is {ticker}."
                                + "\nUse only supplied findings. "
                                "Do not invent missing data."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Company: {company}\nTicker: {ticker}\n"
                                f"Request: {request}\nFindings:\n"
                                + json.dumps(
                                    findings,
                                    indent=2,
                                    ensure_ascii=False,
                                    default=str,
                                )
                            ),
                        },
                    ]

                    report_response = ollama.chat(
                        model=MODEL,
                        messages=report_messages,
                        options={"temperature": 0},
                    )

                    report = field(
                        field(report_response, "message"),
                        "content",
                        "",
                    ) or "Report generation failed."

                    print("\n" + "=" * 55)
                    print("FINAL RESEARCH REPORT")
                    print("=" * 55)
                    print(report)

                    try:
                        saved = await session.call_tool(
                            "save_research_report",
                            arguments={
                                "ticker": ticker,
                                "report": report,
                            },
                        )

                        print("\nSave status:")
                        print(tool_text(saved))

                    except Exception as exc:
                        print(f"Could not save report: {exc}")

                    print()

    except Exception as exc:
        print(f"\nMCP execution error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
