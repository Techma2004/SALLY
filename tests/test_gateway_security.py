import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core.gateway import web
from core.gateway.gateway import Gateway, _MAX_TRACKED_USERS
from core.gateway.security import RateLimiter


def test_remote_client_rejected_without_api_key():
    client = TestClient(web.app, client=("203.0.113.9", 5000))
    assert client.get("/system/stats").status_code == 403
    assert client.get("/memory/recent").status_code == 403
    assert client.post("/chat", json={"message": "hi"}).status_code == 403


def test_health_stays_public():
    client = TestClient(web.app, client=("203.0.113.9", 5000))
    assert client.get("/health").status_code == 200


def test_loopback_client_allowed():
    client = TestClient(web.app, client=("127.0.0.1", 5000))
    assert client.get("/system/stats").status_code == 200


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(2)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")
    with pytest.raises(HTTPException) as exc:
        limiter.check("1.2.3.4")
    assert exc.value.status_code == 429
    limiter.check("5.6.7.8")  # other clients unaffected


def test_last_results_is_bounded():
    gw = Gateway.__new__(Gateway)
    from collections import OrderedDict
    gw._last_results = OrderedDict()
    for i in range(_MAX_TRACKED_USERS + 50):
        gw._remember_last_result(f"user-{i}", "1")
    assert len(gw._last_results) == _MAX_TRACKED_USERS
    assert "user-0" not in gw._last_results
