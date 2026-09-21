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


def test_image_short_circuits_to_multimodal():
    cfg = dict(CFG)
    cfg["routes"] = {
        "fast": {"base_url": "http://fast-stub", "model": "m-fast"},
        "vision": {"base_url": "http://vision-stub", "model": "m-vision",
                   "multimodal": True},
    }
    st = RouterState(cfg)
    with patch.object(RouterState, "ask_kev") as m:
        dec = st.decide([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": "xxx"}},
            {"type": "text", "text": "what is this?"}]}])
    assert dec["route"] == "vision" and dec["source"] == "image"
    m.assert_not_called()  # kev never consulted for images


def test_lowconf_premium_demoted_to_default():
    st = RouterState(dict(CFG))
    ans = _kev_ok("powerful", complexity=1.2)
    ans["model_route"]["confidence"] = 0.42  # below floor 0.5
    with patch.object(RouterState, "ask_kev", return_value=ans):
        dec = st.decide([{"role": "user", "content": "hey"}])
    assert dec["route"] == "fast"  # default cheap route, not premium


def test_noise_stripped_before_classification():
    st = RouterState(dict(CFG))
    noisy = ("<system-reminder>Long automation context with tools list</system-reminder>"
             "<env>platform: osx</env> hey")
    with patch.object(RouterState, "ask_kev",
                      return_value=_kev_ok("fast")) as m:
        st.decide([{"role": "user", "content": noisy}])
    sent = m.call_args.args[0]
    assert "automation context" not in sent and sent.strip() == "hey"


def test_overflow_reroutes_to_biggest_window():
    cfg = dict(CFG)
    cfg["routes"] = {
        "fast": {"base_url": "http://f", "model": "m", "max_context_tokens": 1000},
        "big": {"base_url": "http://b", "model": "m2", "max_context_tokens": 100000},
    }
    st = RouterState(cfg)
    with patch.object(RouterState, "ask_kev", return_value=_kev_ok("fast")):
        dec = st.decide([{"role": "user", "content": "x" * 20000}])  # ~5k tokens
    assert dec["route"] == "big" and dec["source"] == "overflow"


def test_overflow_respects_multimodal():
    cfg = dict(CFG)
    cfg["routes"] = {
        "fast": {"base_url": "http://f", "model": "m", "max_context_tokens": 1000},
        "bigtext": {"base_url": "http://b", "model": "m2",
                    "max_context_tokens": 100000},
        "vision": {"base_url": "http://v", "model": "m3",
                   "max_context_tokens": 50000, "multimodal": True},
    }
    st = RouterState(cfg)
    with patch.object(RouterState, "ask_kev", return_value=_kev_ok("fast")):
        dec = st.decide([{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": "x" * 100}},
            {"type": "text", "text": "y" * 20000}]}])  # needs mm + big window
    # vision (50k, multimodal) fits; bigtext excluded (not multimodal)
    assert dec["route"] == "vision"


def test_config_env_targets():
    import os
    os.environ["KEV_ROUTER_TARGET_FAST"] = "http://from-env:9999"
    # hermetic: explicit missing path -> DEFAULT_CONFIG (prod yaml in cwd
    # must not shadow this test)
    cfg = load_config(path="/nonexistent-kev-router-test.yaml")
    assert cfg["routes"]["fast"]["base_url"] == "http://from-env:9999"
    del os.environ["KEV_ROUTER_TARGET_FAST"]


def test_join_url_no_double_prefix():
    from kev_router.server import join_url
    # target with /v1 + incoming /v1/chat/completions -> single /v1
    assert (join_url("http://h:8317/v1", "/v1/chat/completions")
            == "http://h:8317/v1/chat/completions")
    # target without /v1 -> plain join
    assert (join_url("http://h:8317", "/v1/chat/completions")
            == "http://h:8317/v1/chat/completions")
    # exact-prefix path (e.g. target .../v1 + path /v1)
    assert join_url("http://h:8317/v1", "/v1") == "http://h:8317/v1"
    # trailing slash normalized
    assert (join_url("http://h:8317/v1/", "/v1/messages")
            == "http://h:8317/v1/messages")


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
