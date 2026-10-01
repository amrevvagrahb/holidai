import json
import os
import sqlite3
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from classes import PlannerState, PlanVerdict
from helpers import logger, tokens_used, trim_history
from prompts import build_classifier_prompt, build_system_prompt
from tools import fetch_attractions, fetch_coordinates, fetch_events, fetch_weather

MAX_LLM_CALL_COUNT = 5
MAX_CLARIFY_COUNT = 3
MAX_REVISION_COUNT = 3
MAX_THREAD_TOKENS = 50000

turn_reset = {
    "llm_call_count": 0,
    "revision_count": 0,
    "clarify_count": 0,
    "has_draft": False,
    "approval_decision": "",
}


def validate_env_vars(required_env_vars):
    missing = [v for v in required_env_vars if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            textwrap.dedent(
                f"""
                Missing required environment variable(s): {", ".join(missing)}.
                Add them to your .env file before running.
                """
            )
        )


def begin_turn(state: PlannerState) -> dict:
    return dict(turn_reset)


def make_call_ai(llm):
    def call_ai(state: PlannerState) -> dict:
        messages = [SystemMessage(content=build_system_prompt())] + trim_history(
            state["messages"]
        )
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    return call_ai


def clarify(state: PlannerState) -> dict:
    messages = state["messages"]
    last_AIm_idx = next(
        i
        for i in range(len(messages) - 1, -1, -1)
        if isinstance(messages[i], AIMessage) and messages[i].tool_calls
    )
    recent_messages = messages[last_AIm_idx:]

    candidates = []

    for m in recent_messages:
        if isinstance(m, ToolMessage) and m.name == "fetch_coordinates":
            parsed = json.loads(m.content) if isinstance(m.content, str) else m.content
            if isinstance(parsed, dict) and parsed.get("success"):
                candidates = parsed.get("data", [])

    new_clarify_count = state.get("clarify_count", 0) + 1

    if new_clarify_count > MAX_CLARIFY_COUNT:
        return {
            "messages": [
                AIMessage(
                    content=(
                        textwrap.dedent(
                            """
                            Sorry i couldn't narrow down the location after a few tries.
                            please try again.
                            """
                        )
                    )
                )
            ],
            "clarify_count": new_clarify_count,
        }

    user_answer = interrupt(
        {
            "type": "clarify",
            "question": "Which place did you mean?",
            "candidates": candidates,
        }
    )

    chosen = candidates[int(user_answer)]

    return {
        "messages": [
            HumanMessage(
                content=f"I chose: {chosen['name']}, {chosen.get('region') or ''}, {chosen['country']}, lat={chosen['latitude']}, long={chosen['longitude']}."
            )
        ],
        "clarify_count": new_clarify_count,
    }


def make_classify(llm):
    def classify(state: PlannerState) -> dict:

        messages = state["messages"]
        last_message = messages[-1]

        last_request_m_idx = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], HumanMessage)
            ),
            None,
        )

        current_request_messages = (
            messages[last_request_m_idx:]
            if last_request_m_idx is not None
            else messages
        )

        tool_failures = []
        for i in range(len(current_request_messages) - 2, -1, -1):
            m = current_request_messages[i]
            if isinstance(m, ToolMessage):
                try:
                    parsed = (
                        json.loads(m.content)
                        if isinstance(m.content, str)
                        else m.content
                    )
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("success") is False:
                    tool_failures.append({"tool": m.name, "error": parsed.get("data")})

        prompt = build_classifier_prompt(
            messages[last_request_m_idx].content, last_message.content, tool_failures
        )

        status = "not_ready"
        tokens = 0
        try:
            result = llm.invoke(prompt)
            raw_usage = getattr(result["raw"], "usage_metadata", None) or {}
            tokens = raw_usage.get("total_tokens", 0)
            if result.get("parsed") is not None:
                status = result["parsed"].status
            else:
                logger.warning(
                    "classifier returned unparsable output: %s",
                    result.get("parsing_error"),
                )
        except Exception:
            logger.exception("classifier call failed, defaulting to not_ready")
        updated_state = {
            "messages": [
                AIMessage(
                    id=last_message.id,
                    content=last_message.content,
                    additional_kwargs={"status": status},
                    usage_metadata=last_message.usage_metadata,
                )
            ],
            "classifier_tokens": state.get("classifier_tokens", 0) + tokens,
        }
        logger.info(f"classification result: {status}")
        if status == "ready":
            updated_state["has_draft"] = True
        return updated_state

    return classify


def failed_gather(state: PlannerState) -> dict:
    over_tokens = tokens_used(state) > MAX_THREAD_TOKENS
    reason = (
        "This conversation has crossed its token budget. please start a new conversation."
        if over_tokens
        else "Number of llm calls crossed limit."
    )
    last_message = state["messages"][-1]
    stubs = [
        ToolMessage(
            content=json.dumps(
                {
                    "success": False,
                    "error_type": "limit",
                    "data": f"Tool did not run because {reason}.",
                }
            ),
            tool_call_id=tc["id"],
        )
        for tc in getattr(last_message, "tool_calls", [])
    ]

    if state.get("has_draft"):
        last_draft = next(
            (
                m.content
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage)
                and m.additional_kwargs.get("status") == "ready"
            ),
            None,
        )
        text = f"Sorry, that request could not be fulfilled because {reason}:\n\n {last_draft}."
    else:
        text = f"Sorry, first draft could not be created because {reason}."

    return {"messages": stubs + [AIMessage(content=text)]}


def request_approval(state: PlannerState) -> dict:
    draft = state["messages"][-1].content
    user_decision = interrupt(
        {
            "type": "approval",
            "question": "Heres the draft. Approve it?",
            "draft": draft,
        }
    )
    if user_decision.get("action") == "approve":
        return {"approval_decision": "approved"}
    new_revision_count_will_be = state.get("revision_count", 0) + 1
    if new_revision_count_will_be > MAX_REVISION_COUNT:
        return {
            "approval_decision": "out_of_revisions",
            "messages": [
                AIMessage(
                    content=f"Sorry, that change could not be fulfilled because you ran out of revision count crossed limit. Here is the last draft:\n\n{draft}."
                )
            ],
        }
    else:
        return {
            "approval_decision": "revise",
            "messages": [
                HumanMessage(
                    content=f"please revise: {user_decision.get('feedback', '')}"
                )
            ],
            "revision_count": new_revision_count_will_be,
            "llm_call_count": 0,
        }


def cleanup(state: PlannerState) -> dict:
    return dict(turn_reset)


def route_after_ai(state: PlannerState) -> str:
    decision = tools_condition(state)
    over_calls = state["llm_call_count"] > MAX_LLM_CALL_COUNT
    over_tokens = tokens_used(state) >= MAX_THREAD_TOKENS
    if decision == "tools" and (over_calls or over_tokens):
        return "failed_to_gather"
    return decision


def route_after_classify(state: PlannerState) -> str:
    status = state["messages"][-1].additional_kwargs.get("status")
    return "request_approval" if status == "ready" else END


def route_after_approval(state: PlannerState) -> str:
    return "call_ai" if state["approval_decision"] == "revise" else "cleanup"


def route_after_tools(state: PlannerState) -> str:
    messages = state["messages"]
    last_AIm_idx = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], AIMessage) and messages[i].tool_calls
        ),
        None,
    )

    last_AIm = messages[last_AIm_idx:]

    for m in last_AIm:
        if isinstance(m, ToolMessage) and m.name == "fetch_coordinates":
            try:
                parsed = (
                    json.loads(m.content) if isinstance(m.content, str) else m.content
                )
            except (json.JSONDecodeError, TypeError):
                continue
            if (
                isinstance(parsed, dict)
                and parsed.get("success")
                and len(parsed.get("data", [])) > 1
            ):
                return "clarify"

    return "call_ai"


def route_after_clarify(state: PlannerState) -> str:
    if state.get("clarify_count", 0) > MAX_CLARIFY_COUNT:
        return "cleanup"
    return "call_ai"


def create_agent(tool_llm=None, classifier_llm=None, db_path="data/planner.db"):
    load_dotenv()

    required_env_vars = ["OPENAI_API_KEY", "TICKETMASTER_API_KEY"]
    validate_env_vars(required_env_vars)

    tools = [fetch_weather, fetch_events, fetch_coordinates, fetch_attractions]
    tool_node = ToolNode(tools=tools)

    if tool_llm is None:
        tool_llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, max_retries=3
        ).bind_tools(tools)
    if classifier_llm is None:
        classifier_llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0, max_retries=3
        ).with_structured_output(PlanVerdict, include_raw=True)

    graph = StateGraph(PlannerState)
    graph.add_node("begin_turn", begin_turn)
    graph.add_node("classify", make_classify(classifier_llm))
    graph.add_node("call_ai", make_call_ai(tool_llm))
    graph.add_node("tools", tool_node)
    graph.add_node("clarify", clarify)

    graph.add_node("failed_gather", failed_gather)
    graph.add_node("request_approval", request_approval)
    graph.add_node("cleanup", cleanup)

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "call_ai")
    graph.add_conditional_edges(
        "call_ai",
        route_after_ai,
        {"tools": "tools", END: "classify", "failed_to_gather": "failed_gather"},
    )
    graph.add_conditional_edges(
        "request_approval",
        route_after_approval,
        {"cleanup": "cleanup", "call_ai": "call_ai"},
    )
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"request_approval": "request_approval", END: "cleanup"},
    )

    graph.add_conditional_edges(
        "tools", route_after_tools, {"clarify": "clarify", "call_ai": "call_ai"}
    )

    graph.add_conditional_edges(
        "clarify", route_after_clarify, {"cleanup": "cleanup", "call_ai": "call_ai"}
    )

    graph.add_edge("failed_gather", "cleanup")
    graph.add_edge("cleanup", END)

    if db_path != ":memory:":
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    agent = graph.compile(checkpointer=checkpointer)
    return agent
