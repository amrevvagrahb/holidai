import uuid

import openai
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent import MAX_REVISION_COUNT, MAX_THREAD_TOKENS, create_agent
from helpers import logger, tokens_used


@st.cache_resource
def get_agent():
    return create_agent()


try:
    agent = get_agent()
except RuntimeError as e:
    st.error(f"Configuration problem: {e}")
    st.stop()


def init_states() -> None:
    defaults = {
        "messages": [],
        "awaiting_approval": False,
        "awaiting_clarify": False,
        "revision_count": 0,
        "thread_tokens": 0,
        "clarify_candidates": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = st.query_params.get("thread") or str(uuid.uuid4())
        st.query_params["thread"] = st.session_state.thread_id


def describe_step(node, update) -> str | None:
    NODE_DESCRIPTIONS = {"tools": "tool ran", "classify": "classification done"}
    if node == "call_ai":
        msg = update["messages"][-1]
        if msg.tool_calls:
            names = ", ".join(tc["name"] for tc in msg.tool_calls)
            return f"Deciding to call: {names}"

    if node in NODE_DESCRIPTIONS:
        return NODE_DESCRIPTIONS[node]

    return None


def run_graph(payload, config, is_approval_resume=False) -> None:
    try:
        with st.status("Working on your trip....", expanded=True) as status:
            for chunk in agent.stream(payload, config=config, stream_mode="updates"):
                for node, update in chunk.items():
                    text = describe_step(node, update)
                    if text:
                        st.write(text)
            status.update(label="Done", state="complete", expanded=False)
        snapshot = agent.get_state(config)
        result = snapshot.values

        st.session_state.thread_tokens = tokens_used(result)

        if snapshot.next:
            interrupt_value = snapshot.tasks[0].interrupts[0].value
            if interrupt_value["type"] == "approval":
                st.session_state.awaiting_approval = True
                st.session_state.revision_count = result.get("revision_count", 0)
                draft = snapshot.tasks[0].interrupts[0].value["draft"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": draft}
                )
            else:
                st.session_state.awaiting_clarify = True
                st.session_state.awaiting_approval = False
                st.session_state.clarify_candidates = interrupt_value.get(
                    "candidates", []
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{interrupt_value['question']}",
                    }
                )
        else:
            st.session_state.awaiting_approval = False
            st.session_state.awaiting_clarify = False
            st.session_state.clarify_candidates = []
            if not is_approval_resume:
                last = result["messages"][-1]
                st.session_state.messages.append(
                    {"role": "assistant", "content": last.content}
                )
    except openai.RateLimitError as e:
        logger.exception(
            f"OpenAI rate limit/quota error for thread {st.session_state.thread_id}"
        )
        if getattr(e, "code", None) == "insufficient_quota":
            m = "This API budget ran out. Please check back later."
        else:
            m = "The model is busy right now. Please try again in a minute."
        st.session_state.messages.append({"role": "assistant", "content": m})

    except Exception:
        logger.exception(f"agent.stream failed for thread {st.session_state.thread_id}")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Sorry, something went wrong. Please try again.",
            }
        )


def restore_thread(config: dict) -> None:
    if st.session_state.messages:
        return
    snapshot = agent.get_state(config)
    for m in snapshot.values.get("messages", []):
        if isinstance(m, HumanMessage):
            st.session_state.messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and not m.tool_calls and m.content:
            st.session_state.messages.append(
                {"role": "assistant", "content": m.content}
            )
    if snapshot.next:
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        if interrupt_value["type"] == "approval":
            st.session_state.awaiting_approval = True
            st.session_state.awaiting_clarify = False
            st.session_state.revision_count = snapshot.values.get("revision_count", 0)
        else:
            st.session_state.awaiting_clarify = True
            st.session_state.awaiting_approval = False
            st.session_state.clarify_candidates = interrupt_value.get("candidates", [])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"{interrupt_value['question']}",
                }
            )
    st.session_state.thread_tokens = (
        tokens_used(snapshot.values) if snapshot.values else 0
    )


st.set_page_config(page_title="Holidai", page_icon="🏖️")
st.title("Holidai")
if st.sidebar.button("Reset chat"):
    for key in (
        "messages",
        "awaiting_approval",
        "revision_count",
        "thread_tokens",
        "thread_id",
        "awaiting_clarify",
        "clarify_candidates",
    ):
        st.session_state.pop(key, None)
    st.query_params.pop("thread", None)
    st.rerun()

init_states()
restore_thread({"configurable": {"thread_id": st.session_state.thread_id}})

st.sidebar.divider()
st.sidebar.metric(
    label="tokens used", value=f"{st.session_state.thread_tokens}/{MAX_THREAD_TOKENS}"
)
st.sidebar.metric(
    label="estimated cost", value=f"${st.session_state.thread_tokens * 0.00000015:.4f}"
)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


if st.session_state.awaiting_approval:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve"):
            run_graph(
                Command(resume={"action": "approve"}),
                {"configurable": {"thread_id": st.session_state.thread_id}},
                is_approval_resume=True,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": "Plan approved ✅"}
            )
            st.rerun()
    with col2:
        st.write(
            f"revision count: {st.session_state.revision_count}/{MAX_REVISION_COUNT}"
        )

    feedback = st.chat_input("Request change:")
    if feedback:
        st.session_state.messages.append({"role": "user", "content": feedback})
        run_graph(
            Command(resume={"action": "revise", "feedback": feedback}),
            {"configurable": {"thread_id": st.session_state.thread_id}},
        )
        st.rerun()


elif st.session_state.awaiting_clarify:
    for i, c in enumerate(st.session_state.clarify_candidates):
        parts = [c.get("name"), c.get("region"), c.get("country")]
        label = ", ".join(p for p in parts if p)
        if st.button(label, key=f"clarify_{i}"):
            st.session_state.messages.append({"role": "user", "content": label})
            run_graph(
                Command(resume=str(i)),
                {"configurable": {"thread_id": st.session_state.thread_id}},
            )
            st.rerun()


else:
    user_input = st.chat_input()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        payload = {"messages": [HumanMessage(user_input)]}

        run_graph(payload, {"configurable": {"thread_id": st.session_state.thread_id}})
        st.rerun()
