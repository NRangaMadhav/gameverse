import json
import logging
import os
import re
import sys
from functools import lru_cache
from typing import cast

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.models import ResearchState
from app.settings import PROJECT_ROOT, nvidia_request_timeout, required_setting
from app.vector_store import retrieve

logger = logging.getLogger(__name__)

def _chat_model() -> ChatNVIDIA:
    required_setting("NVIDIA_API_KEY")
    return ChatNVIDIA(
        model=os.getenv("NVIDIA_LLM_MODEL", "google/diffusiongemma-26b-a4b-it"),
        timeout=nvidia_request_timeout(),
        temperature=0,
        max_completion_tokens=128,
    )


def _decode_planner_plan(content: object) -> dict[str, bool] | None:
    text = str(content).strip()
    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            plan = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(plan, dict)
            and isinstance(plan.get("local"), bool)
            and isinstance(plan.get("web"), bool)
            and (plan["local"] or plan["web"])
        ):
            return {"local": plan["local"], "web": plan["web"]}
    return None


def _planner(state: ResearchState) -> dict[str, object]:
    model = _chat_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Return exactly one JSON object and nothing else. Do not include markdown, "
                    "explanations, or reasoning. Schema: "
                    '{"local": true, "web": true}. '
                    "Set local true to search the indexed India gaming reports when relevant. "
                    "Set web true for current facts, external verification, or broader context. "
                    "At least one value must be true."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    plan = _decode_planner_plan(response.content)
    if plan is None:
        logger.warning(
            "Planner response was not valid routing JSON; using conservative local + web routing."
        )
        plan = {"local": True, "web": True}
    return {
        "plan": plan,
        "local_evidence": [],
        "web_evidence": "",
        "local_citations": [],
        "web_citations": [],
    }


def _retrieve_local(state: ResearchState) -> dict[str, object]:
    if not state["plan"]["local"]:
        return {"local_evidence": [], "local_citations": []}
    documents = retrieve(state["question"])
    evidence: list[str] = []
    citations: list[dict[str, str]] = []
    for index, document in enumerate(documents, start=1):
        citation_id = f"L{index}"
        source = str(document.metadata.get("source", "Local report"))
        page = str(document.metadata.get("page", "?"))
        kind = str(document.metadata.get("kind", "text"))
        evidence.append(
            f"[{citation_id}] {source}, page {page} ({kind}):\n{document.page_content}"
        )
        citations.append(
            {"id": citation_id, "title": f"{source} — page {page}", "url": ""}
        )
    return {"local_evidence": evidence, "local_citations": citations}


async def _call_mcp_research(question: str) -> dict[str, object]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("parallel_web_research", {"question": question})
    if result.isError:
        details = "\n".join(
            str(getattr(content, "text", "")) for content in result.content
        ).strip()
        raise RuntimeError(f"MCP web-research tool failed: {details or 'unknown tool error'}")
    text = "\n".join(str(getattr(content, "text", "")) for content in result.content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MCP web-research tool returned malformed JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("MCP web-research tool returned an unexpected response.")
    return cast(dict[str, object], payload)


async def _research_web(state: ResearchState) -> dict[str, object]:
    if not state["plan"]["web"]:
        return {"web_evidence": "", "web_citations": []}
    payload = await _call_mcp_research(state["question"])
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Parallel returned no usable web results.")
    evidence: list[str] = []
    citations: list[dict[str, str]] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        citation_id = f"W{index}"
        title = str(item.get("title") or "Web result")
        url = str(item.get("url") or "")
        excerpt = str(item.get("excerpt") or "")
        evidence.append(f"[{citation_id}] {title} ({url}):\n{excerpt}")
        citations.append({"id": citation_id, "title": title, "url": url})
    return {"web_evidence": "\n\n".join(evidence), "web_citations": citations}


def _synthesize(state: ResearchState) -> dict[str, object]:
    evidence_sections = []
    if state.get("local_evidence"):
        evidence_sections.append("LOCAL REPORTS:\n" + "\n\n".join(state["local_evidence"]))
    if state.get("web_evidence"):
        evidence_sections.append("LIVE WEB RESEARCH:\n" + state["web_evidence"])
    evidence = "\n\n".join(evidence_sections) or "No evidence sources were returned."
    response = _chat_model().invoke(
        [
            SystemMessage(
                content=(
                    "Answer the user's gaming-industry research question using only the supplied "
                    "evidence. Clearly distinguish local report findings from current web findings. "
                    "Resolve conflicts by explaining differences in dates or scope; do not invent "
                    "facts. Cite every factual claim with the supplied IDs such as [L1] or [W1]. "
                    "If evidence is insufficient, say so."
                )
            ),
            HumanMessage(
                content=f"Question: {state['question']}\n\nEvidence:\n{evidence}"
            ),
        ]
    )
    answer = str(response.content).strip()
    if not answer:
        raise RuntimeError("NVIDIA returned an empty research answer.")
    citations = list(state.get("local_citations", [])) + list(
        state.get("web_citations", [])
    )
    return {"answer": answer, "citations": citations}


@lru_cache(maxsize=1)
def get_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", _planner)
    graph.add_node("retriever", _retrieve_local)
    graph.add_node("web_research", _research_web)
    graph.add_node("synthesis", _synthesize)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("planner", "web_research")
    graph.add_edge(["retriever", "web_research"], "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


async def research(question: str) -> dict[str, object]:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Enter a research question before submitting.")
    result = await get_research_graph().ainvoke({"question": cleaned_question})
    return {
        "question": cleaned_question,
        "answer": result["answer"],
        "citations": result.get("citations", []),
    }
