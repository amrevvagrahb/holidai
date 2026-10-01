from datetime import date, timedelta

from helpers import check_dates


def test_check_dates_1():
    today = date.today()
    sd = today.isoformat()
    ed = (today + timedelta(days=3)).isoformat()
    assert check_dates(sd, ed) is None


def test_check_dates_2():
    result = check_dates("23-09-2026", "26-09-2026")
    assert result is not None
    assert "format" in result


def test_check_dates_3():
    today = date.today()
    sd = (today - timedelta(days=1)).isoformat()
    ed = today.isoformat()
    result = check_dates(sd, ed)
    assert result is not None
    assert "past" in result


def test_check_dates_4():
    today = date.today()
    sd = (today + timedelta(days=5)).isoformat()
    ed = (today + timedelta(days=2)).isoformat()
    result = check_dates(sd, ed)
    assert result == "end_date is before start_date"


def test_check_dates_5():
    today = date.today()
    sd = today.isoformat()
    ed = (today + timedelta(days=20)).isoformat()
    result = check_dates(sd, ed, max_days_ahead=16)
    assert result is not None
    assert "days ahead" in result
