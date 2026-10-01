from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent import route_after_ai, route_after_classify


def test_route_after_classify_1():
    current_state = {
        "messages": [
            AIMessage(
                content="heres the plan: swim in the sea",
                additional_kwargs={"status": "ready"},
            )
        ]
    }
    result = route_after_classify(current_state)
    assert result == "request_approval"


def test_route_after_classify_2():
    current_state = {
        "messages": [
            AIMessage(
                content="heres the plan: swim in the sea",
                additional_kwargs={"status": "not_ready"},
            )
        ]
    }
    result = route_after_classify(current_state)
    assert result == END


def test_route_after_classify_3():
    current_state = {
        "messages": [
            AIMessage(
                content="there was some problem gathering info",
                additional_kwargs={"status": "not_ready"},
            )
        ]
    }
    result = route_after_classify(current_state)
    assert result == END


def test_route_after_ai_1():
    current_state = {
        "messages": [
            AIMessage(content="heres the plan: swim in the sea", tool_calls=[])
        ],
        "llm_call_count": 1,
    }
    result = route_after_ai(current_state)
    assert result == END


def test_route_after_ai_2():
    current_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch_weather", "args": {}, "id": "1"}],
            )
        ],
        "llm_call_count": 1,
    }
    result = route_after_ai(current_state)
    assert result == "tools"


def test_route_after_ai_3():
    current_state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch_weather", "args": {}, "id": "1"}],
            )
        ],
        "llm_call_count": 6,
    }
    result = route_after_ai(current_state)
    assert result == "failed_to_gather"
