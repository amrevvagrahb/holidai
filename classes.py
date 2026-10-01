from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanVerdict(BaseModel):
    status: Literal["ready", "not_ready"] = Field(
        description=(
            "'ready' if response is a vacation plan with any tool failures being clearly disclosed. "
            "'not_ready' for everything else"
        )
    )


class PlannerState(TypedDict):
    messages: Annotated[list, add_messages]
    llm_call_count: int
    revision_count: int
    clarify_count: int
    has_draft: bool
    approval_decision: str
    classifier_tokens: int
