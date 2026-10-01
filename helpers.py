import functools
import logging
import os
from datetime import date, datetime

import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def tokens_used(state) -> int:
    total = state.get("classifier_tokens", 0)
    for m in state["messages"]:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            total += usage["total_tokens"]
    return total


def scrub(text: str) -> str:
    key = os.getenv("TICKETMASTER_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text


def check_dates(
    start_date: str, end_date: str, max_days_ahead: int | None = None
) -> str | None:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return "Dates must be in YYYY-MM-DD format, for example 2026-09-20."

    today = date.today()
    if start < today:
        return f"start_date {start_date} is in the past. today is {today}"
    if end < start:
        return "end_date is before start_date"
    if max_days_ahead is not None and (end - today).days > max_days_ahead:
        return f"Forecast only covers about {max_days_ahead} days ahead. Trip is too far out."

    return None


def handle_api_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            logger.warning("%s HTTP %s: %s", func.__name__, status, scrub(str(e)))
            return {
                "success": False,
                "error_type": "api_error",
                "data": f"api returned http {status}",
                "status_code": status,
            }

        except requests.RequestException as e:
            logger.warning("%s network error: %s", func.__name__, scrub(str(e)))
            return {
                "success": False,
                "error_type": "network_error",
                "data": "network error: could not reach the service",
            }

        except (KeyError, TypeError, IndexError) as e:
            logger.warning("%s bad response shape: %s", func.__name__, scrub(str(e)))
            return {
                "success": False,
                "error_type": "unexpected_data",
                "data": f"unexpected response shape from {func.__name__}",
            }

        except Exception as e:
            logger.warning("%s unexpected failure: %s", func.__name__, scrub(str(e)))
            return {
                "success": False,
                "error_type": "unknown",
                "data": "unexpected error",
            }

    return wrapper


def make_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


session = make_session()


def trim_history(messages: list) -> list:
    last_Hm_idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    old = messages[:last_Hm_idx]
    current = messages[last_Hm_idx:]

    kept_old = [
        m
        for m in old
        if not isinstance(m, ToolMessage)
        and not (isinstance(m, AIMessage) and m.tool_calls)
    ]

    return kept_old + current
