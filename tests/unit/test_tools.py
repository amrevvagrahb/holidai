from fakes import FakeAPIResponse

from tools import fetch_coordinates, fetch_events, fetch_weather


def test_fetch_coordinates_1(mocker):
    mocker.patch(
        "helpers.session.get",
        return_value=FakeAPIResponse(
            {
                "results": [
                    {
                        "name": "Testville",
                        "admin1": "Test Region",
                        "country": "Testland",
                        "latitude": 10.0,
                        "longitude": 20.0,
                    }
                ]
            }
        ),
    )

    result = fetch_coordinates.invoke({"area": "testville_success"})
    assert result["success"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["region"] == "Test Region"


def test_fetch_coordinates_2(mocker):
    mocker.patch("helpers.session.get", return_value=FakeAPIResponse({"results": []}))

    result = fetch_coordinates.invoke({"area": "testville_success_2"})
    assert result["success"] is False
    assert result["data"] == "location not found"


from datetime import date, timedelta

today = date.today()
sd = today.isoformat()
ed = (today + timedelta(days=3)).isoformat()


def test_fetch_weather_1(mocker):
    mocker.patch(
        "helpers.session.get",
        return_value=FakeAPIResponse(
            {
                "daily": {
                    "time": [(today + timedelta(days=i)).isoformat() for i in range(3)],
                    "temperature_2m_max": [30.5, 31.0, 20.0],
                    "temperature_2m_min": [22.1, 21.8, 15.0],
                    "precipitation_probability_max": [10, 40, 50],
                }
            }
        ),
    )

    result = fetch_weather.invoke({"sd": sd, "ed": ed, "lat": 10.0, "long": 15.0})
    assert result["success"] is True
    assert len(result["data"]) == 3
    assert result["data"][0]["high"][-2:] == "°C"
    assert result["data"][0]["precipitation_probability"][-1] == "%"
    assert result["data"][0]["high"] == "30.5°C"


def test_fetch_weather_2():
    result = fetch_weather.invoke(
        {
            "sd": date.today().isoformat(),
            "ed": (date.today() + timedelta(days=-3)).isoformat(),
            "lat": 1.0,
            "long": 1.5,
        }
    )
    assert result["success"] is False
    assert result["data"] == "end_date is before start_date"


def test_fetch_events_1(mocker):
    mocker.patch("helpers.session.get", return_value=FakeAPIResponse({}))
    today = date.today()
    sd = today.isoformat()
    ed = (today + timedelta(days=4)).isoformat()

    result = fetch_events.invoke({"sd": sd, "ed": ed, "lat": 100.0, "long": 120.0})
    assert result["success"] is True
    assert (
        result["data"]
        == "search worked but no events were found for these dates and that area"
    )


def test_fetch_events_2(mocker):
    mocker.patch(
        "helpers.session.get",
        return_value=FakeAPIResponse(
            {
                "_embedded": {
                    "events": [
                        {
                            "name": "Test Concert",
                            "dates": {
                                "start": {
                                    "localDate": "2026-10-01",
                                    "localTime": "19:00",
                                },
                                "status": {"code": "onsale"},
                            },
                            "_embedded": {
                                "venues": [
                                    {
                                        "name": "Test Arena",
                                        "city": {"name": "Testville"},
                                        "state": {"stateCode": "TS"},
                                    }
                                ]
                            },
                            "priceRanges": [{"min": 50, "max": 100, "currency": "USD"}],
                            "url": "https://example.com/event",
                        }
                    ]
                }
            }
        ),
    )
    today = date.today()
    sd = today.isoformat()
    ed = (today + timedelta(days=4)).isoformat()

    result = fetch_events.invoke({"sd": sd, "ed": ed, "lat": 100.0, "long": 120.0})
    assert result["success"] is True
    assert len(result["data"]) == 1
    assert result["data"][0]["name"] == "Test Concert"
    assert result["data"][0]["multi_venue"] is False


from tools import fetch_attractions


def test_fetch_attractions_1(mocker):
    mocker.patch(
        "helpers.session.get",
        return_value=FakeAPIResponse({"query": {"geosearch": []}}),
    )

    result = fetch_attractions.invoke({"lat": 15.0, "long": 30.0})
    assert result["success"] is True
    assert len(result["data"]) == 0


def test_fetch_attractions_2(mocker):
    mocker.patch(
        "helpers.session.get",
        return_value=FakeAPIResponse(
            {"query": {"geosearch": [{"title": "test landmark", "dist": 34.2}]}}
        ),
    )

    result = fetch_attractions.invoke({"lat": 15.0, "long": 30.0})
    assert result["success"] is True
    assert result["data"][0]["distance_m"] == 34
