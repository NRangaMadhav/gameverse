import logging
import os

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from app.settings import PROJECT_ROOT, required_setting

load_dotenv(PROJECT_ROOT / ".env")
logging.basicConfig(level=logging.INFO)
mcp = FastMCP("GameVerse India research tools")


@mcp.tool()
def parallel_web_research(question: str) -> dict[str, object]:
    """Search current web sources for gaming-industry research relevant to a question."""
    endpoint = os.getenv(
        "PARALLEL_SEARCH_URL", "https://api.parallel.ai/v1/search"
    ).strip()
    response = requests.post(
        endpoint,
        headers={
            "x-api-key": required_setting("PARALLEL_API_KEY"),
            "Content-Type": "application/json",
        },
        json={
            "objective": question,
            "search_queries": [question],
            "mode": "fast",
        },
        timeout=45,
    )
    if not response.ok:
        raise RuntimeError(
            f"Parallel web research failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Parallel returned an unexpected response format.")

    raw_results = payload.get("results") or payload.get("search_results") or []
    results: list[dict[str, str]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                results.append(
                    {
                        "title": str(item.get("title") or "Web result"),
                        "url": str(item.get("url") or item.get("link") or ""),
                        "excerpt": str(
                            "\n".join(str(excerpt) for excerpt in item.get("excerpts", []))
                            if isinstance(item.get("excerpts"), list)
                            else item.get("excerpt")
                            or item.get("snippet")
                            or item.get("description")
                            or ""
                        ),
                    }
                )
    return {"query": question, "results": results}


if __name__ == "__main__":
    mcp.run()
