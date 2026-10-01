from datetime import date, timedelta
from unittest.mock import Mock

import requests
from langchain_core.messages import AIMessage

from classes import PlanVerdict


class FakeAPIResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self.json_data


def make_fake_tool_llm_1():
    today = date.today()
    sd = today.isoformat()
    ed = (today + timedelta(days=5)).isoformat()
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_coordinates", "id": "call_1", "args": {"area": "tokyo"}}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_weather",
                    "id": "call_2",
                    "args": {"sd": sd, "ed": ed, "lat": 34.05, "long": -118.24},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_events",
                    "id": "call_3",
                    "args": {"lat": 34.05, "long": -118.24, "sd": sd, "ed": ed},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_attractions",
                    "id": "call_4",
                    "args": {"lat": 34.05, "long": -118.24},
                }
            ],
        ),
        AIMessage(content="heres your plan for tokyo"),
    ]
    return llm


def make_fake_classifier_llm_1():
    llm = Mock()
    llm.invoke.side_effect = [
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        }
    ]
    return llm


def fake_get_1(url, *args, **kwargs):
    if "geocoding-api" in url:
        return FakeAPIResponse(
            {
                "results": [
                    {
                        "name": "tokyo",
                        "admin1": "carifornia",
                        "country": "japan",
                        "latitude": 48.85,
                        "longitude": 2.35,
                    }
                ]
            }
        )
    elif "open-meteo.com/v1/forecast" in url:
        return FakeAPIResponse(
            {
                "daily": {
                    "time": ["2026-09-23"],
                    "temperature_2m_max": [25],
                    "temperature_2m_min": [15],
                    "precipitation_probability_max": [10],
                }
            }
        )
    elif "ticketmaster.com" in url:
        return FakeAPIResponse(
            {
                "_embedded": {
                    "events": [
                        {
                            "name": "Fake Concert",
                            "dates": {
                                "start": {
                                    "localDate": "2026-09-23",
                                    "localTime": "19:00",
                                },
                                "status": {"code": "onsale"},
                            },
                            "_embedded": {
                                "venues": [
                                    {
                                        "name": "Fake Arena",
                                        "city": {"name": "Los Angeles"},
                                        "state": {"stateCode": "CA"},
                                    }
                                ]
                            },
                            "priceRanges": [{"min": 50, "max": 150, "currency": "USD"}],
                            "url": "https://example.com/event",
                        }
                    ]
                }
            }
        )
    elif "wikipedia.org" in url:
        return FakeAPIResponse(
            {"query": {"geosearch": [{"title": "Griffith Observatory", "dist": 500}]}}
        )
    raise ValueError(f"unexpected url: {url}")


def fake_get_2(*args, **kwargs):
    return FakeAPIResponse(
        {
            "results": [
                {
                    "name": "seoul",
                    "admin1": "one",
                    "country": "south korea",
                    "latitude": 48.85,
                    "longitude": 2.35,
                },
                {
                    "name": "seoul",
                    "admin1": "two",
                    "country": "south korea",
                    "latitude": 33.66,
                    "longitude": -95.56,
                },
            ]
        }
    )


def make_fake_tool_llm_2():
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_coordinates", "id": "call_1", "args": {"area": "seoul"}}
            ],
        ),
        AIMessage(content="Here's your plan for Paris, France!"),
    ]
    return llm


def make_fake_classifier_llm_3():
    llm = Mock()
    llm.invoke.side_effect = [
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
    ]
    return llm


def make_fake_tool_llm_3():
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "fetch_coordinates", "id": "call_1", "args": {"area": "paris"}}
            ],
        ),
        AIMessage(content="Here's your plan for los angeles!"),
        AIMessage(content="Here's your plan for los angeles with emojis: "),
    ]
    return llm


def make_fake_classifier_llm_4():
    llm = Mock()
    llm.invoke.side_effect = [
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
        {
            "parsed": PlanVerdict(status="ready"),
            "raw": Mock(usage_metadata={"total_tokens": 42}),
        },
    ]
    return llm


def make_fake_tool_llm_4():
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_coordinates",
                    "id": "call_1",
                    "args": {"area": "los angeles"},
                }
            ],
        ),
        AIMessage(content="Here's your plan for los angeles!"),
        AIMessage(content="Here's your plan for los angeles with emojis: "),
        AIMessage(content="Here's your plan for los angeles without emojis: "),
        AIMessage(content="Here's your plan for los angeles with emojis: "),
    ]
    return llm


def make_fake_tool_llm_5():
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_coordinates",
                    "id": "call_1",
                    "args": {"area": "pyongyang"},
                }
            ],
        )
        for _ in range(1, 10)
    ]
    return llm


def make_fake_classifier_llm_6():
    llm = Mock()
    llm.invoke.side_effect = [
        {
            "parsed": None,
            "raw": Mock(usage_metadata={"total_tokens": 42}),
            "parsing_error": ValueError("bad output"),
        }
    ]
    return llm


def make_fake_classifier_llm_7():
    llm = Mock()
    llm.invoke.side_effect = RuntimeError("provider down")
    return llm
