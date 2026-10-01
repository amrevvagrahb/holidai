import os
from functools import lru_cache
from typing import Literal

from langchain_core.tools import tool

from helpers import check_dates, handle_api_errors, logger, session


@lru_cache(maxsize=256)
def _geocode(area: str) -> tuple[dict, ...]:
    response = session.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": area, "count": 5},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    stuff = data.get("results") or []
    return tuple(
        {
            "name": r.get("name"),
            "region": r.get("admin1"),
            "country": r.get("country"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
        }
        for r in stuff
    )


@tool
@handle_api_errors
def fetch_weather(sd: str, ed: str, lat: float, long: float) -> dict:
    """Fetch weather forecast at given co-ordinates from start date to end date."""

    error = check_dates(sd, ed, max_days_ahead=16)
    if error:
        return {"success": False, "error_type": "invalid_input", "data": error}

    params = {
        "timezone": "auto",
        "latitude": lat,
        "longitude": long,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "start_date": sd,
        "end_date": ed,
    }
    url = "https://api.open-meteo.com/v1/forecast"
    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    stuff = data["daily"]
    if not (
        len(stuff["time"])
        == len(stuff["temperature_2m_max"])
        == len(stuff["temperature_2m_min"])
        == len(stuff["precipitation_probability_max"])
    ):
        raise IndexError

    report = []
    for i in range(len(stuff["time"])):
        report.append(
            {
                "date": stuff["time"][i],
                "high": f"{stuff['temperature_2m_max'][i]}°C",
                "low": f"{stuff['temperature_2m_min'][i]}°C",
                "precipitation_probability": f"{stuff['precipitation_probability_max'][i]}%",
            }
        )

    logger.info("fetch_weather succeeded for (%s, %s) %s to %s", lat, long, sd, ed)
    return {"success": True, "data": report}


@tool
@handle_api_errors
def fetch_events(
    lat: float,
    long: float,
    sd: str,
    ed: str,
    category: Literal["music", "arts & theatre", "sports", "film", "miscellaneous"]
    | None = None,
    radius_km: int = 25,
) -> dict:
    """
    Fetch events near a location between two dates.
    Category is optional: one of 'music', 'sports', 'arts & theatre', 'film', 'miscellaneous'.
    """
    error = check_dates(sd, ed)
    if error:
        return {"success": False, "error_type": "invalid_input", "data": error}
    API_KEY = os.getenv("TICKETMASTER_API_KEY")
    url = "https://app.ticketmaster.com/discovery/v2/events.json"

    def get_useful_event_info(e):
        venues = e.get("_embedded", {}).get("venues", [])
        venue = venues[0] if venues else {}
        prices = e.get("priceRanges", [{}])
        price = prices[0] if len(prices) > 0 else {}
        start = e.get("dates", {}).get("start", {})
        return {
            "name": e.get("name"),
            "date": start.get("localDate"),
            "time": start.get("localTime"),
            "venue": venue.get("name"),
            "city": venue.get("city", {}).get("name"),
            "state": venue.get("state", {}).get("stateCode"),
            "price_min": price.get("min"),
            "price_max": price.get("max"),
            "currency": price.get("currency"),
            "status": e.get("dates", {}).get("status", {}).get("code"),
            "url": e.get("url"),
            "multi_venue": len(venues) > 1,
        }

    params = {
        "apikey": API_KEY,
        "size": 10,
        "latlong": f"{lat},{long}",
        "radius": radius_km,
        "localStartEndDateTime": f"{sd}T00:00:00,{ed}T23:59:59",
        "unit": "km",
    }

    if category:
        params["classificationName"] = category

    response = session.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    stuff = data.get("_embedded", {}).get("events", [])
    report = [get_useful_event_info(e) for e in stuff]

    if not report:
        logger.info(
            "fetch_events succeeded, but found no events for (%s, %s) %s to %s",
            lat,
            long,
            sd,
            ed,
        )
        return {
            "success": True,
            "data": "search worked but no events were found for these dates and that area",
        }

    logger.info("fetch_events succeeded, %d event(s) found", len(report))
    return {"success": True, "data": report}


@tool
@handle_api_errors
def fetch_coordinates(area: str) -> dict:
    """Find candidates for a place query. Return up to 5 matches, each with name, region, country, latitude and longitude."""
    candidates = _geocode(area.strip().lower())
    if not candidates:
        logger.info("fetch coordinates succeeded: area not found: %s", area)
        return {
            "success": False,
            "error_type": "not_found",
            "data": "location not found",
        }
    logger.info(
        "fetch_coordinates succeeded: %d candidate(s) for area= %s",
        len(candidates),
        area,
    )
    return {"success": True, "data": candidates}


@tool
@handle_api_errors
def fetch_attractions(lat: float, long: float, radius_km: int = 10) -> dict:
    """Find notable places near a location. Results come from wikipedia."""
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{long}",
        "gsradius": min(radius_km, 10) * 1000,
        "gslimit": 20,
        "format": "json",
    }

    response = session.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers={"User-Agent": "holidai-portfolio-project/0.1"},
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()
    stuff = data["query"]["geosearch"]
    if not stuff:
        return {"success": True, "data": []}

    places = [{"name": r["title"], "distance_m": round(r["dist"])} for r in stuff]
    logger.info("fetch_attractions succeeded, %d place(s)", len(places))
    return {"success": True, "data": places}
