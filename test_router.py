"""Offline tests: kev stubbed, no network, no secrets. Run: python -m pytest test_router.py"""
import json
from unittest.mock import patch

from kev_router.config import load_config
from kev_router.server import RouterState, create_app
from fastapi.testclient import TestClient

CFG = {
    "kev_url": "http://kev-stub/v1/systemone",
    "listen_port": 8323,
    "default_route": "fast",
    "kev_timeout_s": 1.0,
    "cache_ttl_s": 3600,
    "max_state_chars": 6000,
    "complexity_threshold": 3.0,
    "complexity_route": "powerful",
    "routes": {
        "fast": {"base_url": "http://fast-stub", "model": "m-fast",
                 "criteria": "lookups, short edits"},
        "powerful": {"base_url": "http://powerful-stub", "model": "m-power",
                     "criteria": "reasoning, architecture"},
    },
}


def _kev_ok(route="fast", complexity=1.0):
    # shape returned by ask_kev(): the answers dict directly (not the full
    # HTTP response)
    return {
        "model_route": {"type": "choice", "choice": route, "confidence": 0.91},
        "complexity": {"type": "score", "score": complexity, "confidence": 0.8},
    }


def test_fail_open_on_kev_down():
    st = RouterState(dict(CFG))
    dec = st.decide([{"role": "user", "content": "hello"}])
    assert dec["source"] == "fail-open"
    assert dec["route"] == "fast"


def test_kev_choice_wins():
    st = RouterState(dict(CFG))
    with patch.object(RouterState, "ask_kev", return_value=_kev_ok("powerful")):
        dec = st.decide([{"role": "user", "content": "hard problem"}])
    assert dec["route"] == "powerful" and dec["source"] == "kev"


def test_complexity_upgrade():
    st = RouterState(dict(CFG))
    with patch.object(RouterState, "ask_kev",
                      return_value=_kev_ok("fast", complexity=3.2)):
        dec = st.decide([{"role": "user", "content": "big doc summary"}])
    assert dec["route"] == "powerful" and dec["source"] == "kev"


def test_cache_per_conversation():
    st = RouterState(dict(CFG))
    with patch.object(RouterState, "ask_kev",
                      return_value=_kev_ok("powerful")) as m:
        st.decide([{"role": "user", "content": "first"}])
        st.decide([{"role": "user", "content": "first"}, {"role": "user", "content": "more"}])
        st.decide([{"role": "user", "content": "different conversation"}])
    assert m.call_count == 2  # 3rd request = new conversation


def test_anthropic_content_blocks():
    st = RouterState(dict(CFG))
    with patch.object(RouterState, "ask_kev", return_value=_kev_ok("fast")):
        dec = st.decide([{"role": "user", "content":
                          [{"type": "text", "text": "blocks format"}]}])
    assert dec["route"] == "fast"


def test_config_env_targets():
    import os
    os.environ["KEV_ROUTER_TARGET_FAST"] = "http://from-env:9999"
    cfg = load_config()
    assert cfg["routes"]["fast"]["base_url"] == "http://from-env:9999"
    del os.environ["KEV_ROUTER_TARGET_FAST"]


def test_e2e_forward_and_header():
    app = create_app(dict(CFG))
    with patch.object(RouterState, "ask_kev", return_value=_kev_ok("fast")):
        c = TestClient(app)
        r = c.post("/v1/chat/completions", json={
            "model": "ignored", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code in (200, 502)  # stub target is unreachable, fine
        assert "x-kev-route" in r.headers
        sent = json.loads(r.headers["x-kev-route"])
        assert sent["route"] == "fast"
