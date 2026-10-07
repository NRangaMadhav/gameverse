from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    plan: dict[str, bool]
    local_evidence: list[str]
    web_evidence: str
    local_citations: list[dict[str, str]]
    web_citations: list[dict[str, str]]
    citations: list[dict[str, str]]
    answer: str
