import os

from fakes import (
    fake_get_1,
    fake_get_2,
    make_fake_classifier_llm_1,
    make_fake_classifier_llm_3,
    make_fake_classifier_llm_4,
    make_fake_classifier_llm_6,
    make_fake_classifier_llm_7,
    make_fake_tool_llm_1,
    make_fake_tool_llm_2,
    make_fake_tool_llm_3,
    make_fake_tool_llm_4,
    make_fake_tool_llm_5,
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import create_agent

os.environ["OPENAI_API_KEY"] = "testkey"
os.environ["TICKETMASTER_API_KEY"] = "testkey"


def test_1(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_1(),
        classifier_llm=make_fake_classifier_llm_1(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-1"}}
    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to los angeles")]},
        config=config,
    )
    snapshot = agent.get_state(config)
    assert snapshot.next
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"
    agent.invoke(Command(resume={"action": "approve"}), config=config)
    final_snapshot = agent.get_state(config)
    assert not final_snapshot.next


def test_2(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_2)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_2(),
        classifier_llm=make_fake_classifier_llm_1(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-2"}}
    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to seoul")]}, config=config
    )
    snapshot = agent.get_state(config)
    assert snapshot.next
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "clarify"
    assert len(interrupt_value["candidates"]) == 2
    agent.invoke(Command(resume="0"), config=config)
    snapshot = agent.get_state(config)
    assert snapshot.next
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"
    agent.invoke(Command(resume={"action": "approve"}), config=config)
    final_snapshot = agent.get_state(config)
    assert not final_snapshot.next


def test_3(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_3(),
        classifier_llm=make_fake_classifier_llm_3(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-3"}}
    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to los angeles")]},
        config=config,
    )
    snapshot = agent.get_state(config)
    assert snapshot.next
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"
    agent.invoke(
        Command(resume={"action": "revise", "feedback": "add emojis"}), config=config
    )
    snapshot2 = agent.get_state(config)
    assert snapshot2.next
    interrupt_value = snapshot2.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"
    agent.invoke(Command(resume={"action": "approve"}), config=config)
    snapshot3 = agent.get_state(config)
    assert not snapshot3.next


def test_4(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_4(),
        classifier_llm=make_fake_classifier_llm_4(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-4"}}

    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to los angeles")]},
        config=config,
    )
    snapshot = agent.get_state(config)
    assert snapshot.next
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"

    agent.invoke(
        Command(resume={"action": "revise", "feedback": "add emojis"}), config=config
    )
    snapshot2 = agent.get_state(config)
    assert snapshot2.next
    interrupt_value = snapshot2.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"

    agent.invoke(
        Command(resume={"action": "revise", "feedback": "remove emojis"}), config=config
    )
    snapshot3 = agent.get_state(config)
    assert snapshot3.next
    interrupt_value = snapshot3.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"

    agent.invoke(
        Command(resume={"action": "revise", "feedback": "add emojis"}), config=config
    )
    snapshot4 = agent.get_state(config)
    assert snapshot4.next
    interrupt_value = snapshot4.tasks[0].interrupts[0].value
    assert interrupt_value["type"] == "approval"

    agent.invoke(
        Command(resume={"action": "revise", "feedback": "add a joke"}), config=config
    )
    snapshot5 = agent.get_state(config)
    assert not snapshot5.next


def test_5(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_5(),
        classifier_llm=make_fake_classifier_llm_1(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-5"}}

    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to pyongyang")]}, config=config
    )
    snapshot = agent.get_state(config)
    assert not snapshot.next


def test_6(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_1(),
        classifier_llm=make_fake_classifier_llm_6(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-6"}}
    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to los angeles")]},
        config=config,
    )
    snapshot = agent.get_state(config)
    assert not snapshot.next


def test_7(mocker):
    mocker.patch("helpers.session.get", side_effect=fake_get_1)
    agent = create_agent(
        tool_llm=make_fake_tool_llm_1(),
        classifier_llm=make_fake_classifier_llm_7(),
        db_path=":memory:",
    )
    config = {"configurable": {"thread_id": "test-thread-classifier-raises"}}
    agent.invoke(
        {"messages": [HumanMessage(content="plan a trip to los angeles")]},
        config=config,
    )
    snapshot = agent.get_state(config)
    assert not snapshot.next
