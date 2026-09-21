"""kev-router server: OpenAI-compatible proxy that routes with kev.

Every incoming chat request is classified by kev (one forward pass, no text
generation) which picks the route; the request is then forwarded with its
original auth headers to the provider behind that route. kev being down never
blocks traffic (fail-open to the default route).
"""
import hashlib
import json
import os
import re
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_kev_api_key, load_config

PASS_HEADERS = ("authorization", "x-api-key", "content-type", "accept",
                "anthropic-version", "x-request-id", "user-agent")


def messages_contains_image(messages: list) -> bool:
    """True if any message carries image content (Anthropic or OpenAI shape)."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in ("image", "image_url"):
                    return True
        elif isinstance(c, str) and "data:image/" in c:
            return True
    return False


def estimate_tokens(messages: list) -> int:
    """Rough token estimate: chars/4 for text + flat cost per image."""
    chars = 0
    images = 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    chars += len(b.get("text", ""))
                elif b.get("type") in ("image", "image_url"):
                    images += 1
    return chars // 4 + images * 1500


class RouterState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cache: dict[str, tuple[dict, float]] = {}
        self.stats = {"routed": 0, "cache_hits": 0, "fail_open": 0,
                      "upgraded": 0, "lowconf_demoted": 0,
                      "overflow_rerouted": 0, "errors": 0}

    # -- kev ---------------------------------------------------------------
    def ask_kev(self, state: str) -> dict | None:
        headers = {"Content-Type": "application/json"}
        key = get_kev_api_key(self.cfg)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            with httpx.Client(timeout=self.cfg["kev_timeout_s"]) as c:
                r = c.post(self.cfg["kev_url"], json={
                    "model": "kev-latest",
                    "state": state,
                    "questions": {
                        "model_route": {  # NB: kev requires criteria as {option: description}
                            "type": "choice",
                            "instructions": (
                                "Which model route should handle this request?"),
                            "criteria": {name: rt.get("criteria", "")
                                         for name, rt in self.cfg["routes"].items()},
                        },
                        "complexity": {
                            "type": "score",
                            "instructions": "Overall task complexity",
                            "criteria": ["trivial", "simple", "moderate", "complex"],
                        },
                    },
                })
            if r.status_code != 200:
                print(f"[kev-router] kev error {r.status_code}: {r.text[:200]}",
                      flush=True)
                return None
            return r.json().get("answers", {})
        except Exception as e:
            print(f"[kev-router] kev unreachable ({e}); fail-open", flush=True)
            return None

    # -- decision ----------------------------------------------------------
    @staticmethod
    def conv_key(messages: list) -> str:
        first = next((m.get("content", "") for m in messages
                      if isinstance(m, dict) and m.get("role") == "user"), "")
        if isinstance(first, list):
            first = " ".join(b.get("text", "") for b in first
                             if isinstance(b, dict))
        return hashlib.sha1(str(first)[:4000].encode()).hexdigest()[:16]

    @staticmethod
    def last_user(messages: list) -> str:
        c = next((m.get("content", "") for m in reversed(messages)
                  if isinstance(m, dict) and m.get("role") == "user"), "")
        if isinstance(c, list):
            c = " ".join(b.get("text", "") for b in c
                         if isinstance(b, dict) and b.get("type") == "text")
        c = str(c)
        # strip Claude Code wrapper noise (system reminders, env blocks) so
        # classification sees the user's actual words
        c = re.sub(r"<system-reminder>.*?</system-reminder>", " ", c, flags=re.S)
        c = re.sub(r"<env>.*?</env>", " ", c, flags=re.S).strip()
        if not c:  # image-only or fully-stripped message
            c = "[non-text content]"
        return c[: int(os.environ.get("KEV_ROUTER_MAX_STATE_CHARS",
                                      str(2000)))]

    def decide(self, messages: list) -> dict:
        key = self.conv_key(messages)
        hit = self.cache.get(key)
        now = time.time()
        if hit:
            # fail-open decisions are cached only briefly so kev is retried
            # quickly instead of poisoning the conversation for a full TTL
            ttl = (self.cfg.get("fail_cache_ttl_s", 45)
                   if hit[0].get("source") == "fail-open"
                   else self.cfg["cache_ttl_s"])
            if now - hit[1] < ttl:
                self.stats["cache_hits"] += 1
                return {**hit[0], "source": "cache"}

        route_name = None
        complexity = None
        conf = 0.0
        source = None
        if messages_contains_image(messages):
            mm = next((n for n, rt in self.cfg["routes"].items()
                       if rt.get("multimodal")), None)
            if mm:  # deterministic: images need the multimodal route
                route_name, conf, source = mm, 1.0, "image"
        if source is None:
            ans = self.ask_kev(self.last_user(messages))
            if ans:
                rc = ans.get("model_route", {})
                name = rc.get("choice")
                if name in self.cfg["routes"]:
                    route_name, conf = name, rc.get("confidence", 0.0)
                complexity = ans.get("complexity", {}).get("score")

        if route_name is None:  # fail-open
            self.stats["fail_open"] += 1
            route_name = self.cfg["default_route"]
            source = "fail-open"
        else:
            self.stats["routed"] += 1
            if source is None:
                source = "kev"
            # low-confidence guard: an unsure kev decision must never gamble
            # on a premium tier (wrapper noise can read as "automation");
            # demote to the default cheap route unless the task is complex
            floor = float(self.cfg.get("lowconf_min_confidence", 0.5))
            if (source == "kev" and conf < floor
                    and route_name != self.cfg["default_route"]
                    and route_name in self.cfg["routes"]
                    and (complexity is None or complexity < self.cfg.get(
                        "complexity_threshold", 3.0))):
                route_name = self.cfg["default_route"]
                self.stats["lowconf_demoted"] += 1
            # complexity override: cheap route but heavy task -> upgrade
            if (route_name != self.cfg["complexity_route"]
                    and complexity is not None
                    and complexity >= self.cfg["complexity_threshold"]
                    and self.cfg["complexity_route"] in self.cfg["routes"]):
                route_name = self.cfg["complexity_route"]
                self.stats["upgraded"] += 1

        route = self.cfg["routes"].get(route_name)
        if route is None:  # config inconsistency guard: never 500
            route_name = next(iter(self.cfg["routes"]))
            route = self.cfg["routes"][route_name]
        # context-window guard: a payload bigger than the chosen model's
        # window would 400 upstream (context overflow). Reroute to the
        # largest declared window, respecting modality (images stay on a
        # multimodal route even when rerouted).
        est = estimate_tokens(messages)
        cap = route.get("max_context_tokens")
        if cap is not None and est > int(cap):
            need_mm = messages_contains_image(messages)

            def fits(n):
                rt = self.cfg["routes"][n]
                c = rt.get("max_context_tokens")
                return c is not None and est <= int(c) and (
                    rt.get("multimodal") or not need_mm)

            candidates = [n for n in self.cfg["routes"] if fits(n)]
            if candidates:
                big = max(candidates, key=lambda n: int(
                    self.cfg["routes"][n]["max_context_tokens"]))
                if big != route_name:
                    route_name = big
                    route = self.cfg["routes"][big]
                    source = "overflow"
                    self.stats["overflow_rerouted"] += 1
        decision = {"route": route_name, "target": route.get("base_url"),
                    "model": route.get("model"), "confidence": round(conf, 2),
                    "complexity": complexity, "source": source}
        self.cache[key] = (decision, now)
        return decision


from urllib.parse import urlparse

def join_url(target: str, path: str) -> str:
    """Join a target base URL with an incoming path, avoiding duplicated
    path prefixes (e.g. target '.../v1' + path '/v1/chat/completions'
    must give '.../v1/chat/completions', not '.../v1/v1/...')."""
    t = target.rstrip("/")
    p = path if path.startswith("/") else "/" + path
    tp = urlparse(t).path.rstrip("/")
    if tp and (p == tp or p.startswith(tp + "/")):
        p = p[len(tp):]
    return t + p

def create_app(cfg: dict | None = None) -> FastAPI:
    st = RouterState(cfg or load_config())
    app = FastAPI(title="kev-router", version="0.1.0",
                  docs_url=None, redoc_url=None)

    async def handle(req: Request, path: str):
        try:
            body = await req.json()
        except Exception:
            return JSONResponse({"error": {"message": "invalid JSON"}},
                                status_code=400)
        messages = body.get("messages") or []
        dec = st.decide(messages)
        target = dec["target"]
        if not target:
            return JSONResponse({"error": {"message": (
                f"route '{dec['route']}' has no target URL. Set the env var "
                "pointing to that provider (see README).")}},
                status_code=503)
        if dec["model"]:
            body["model"] = dec["model"]  # else keep the client's model as-is
        fwd = {k: v for k, v in req.headers.items() if k.lower() in PASS_HEADERS}
        print(f"[kev-router] {dec['route']} conf={dec['confidence']} "
              f"cx={dec['complexity']} src={dec['source']} -> {target} {path}",
              flush=True)
        try:
            if body.get("stream"):
                def gen():
                    with httpx.Client(timeout=600) as c:
                        with c.stream("POST", join_url(target, path), json=body,
                                      headers=fwd) as r:
                            for chunk in r.iter_raw():
                                yield chunk
                return StreamingResponse(gen(), media_type="text/event-stream",
                                         headers={"x-kev-route": json.dumps(dec)})
            async with httpx.AsyncClient(timeout=600) as c:
                r = await c.post(join_url(target, path), json=body, headers=fwd)
            return Response(r.content, status_code=r.status_code,
                            media_type=r.headers.get(
                                "content-type", "application/json"),
                            headers={"x-kev-route": json.dumps(dec)})
        except Exception as e:
            st.stats["errors"] += 1
            return JSONResponse({"error": {"message": f"upstream error: {e}",
                                           "type": "bad_gateway"}},
                                status_code=502,
                                headers={"x-kev-route": json.dumps(dec)})

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        return await handle(req, "/v1/chat/completions")

    @app.post("/v1/messages")
    async def messages_ep(req: Request):
        return await handle(req, "/v1/messages")

    @app.get("/router/health")
    async def health():
        return {"ok": True, "routes": {k: v.get("base_url")
                                       for k, v in st.cfg["routes"].items()},
                "stats": st.stats}

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": name, "object": "model",
                          "owned_by": "kev-router"}
                         for name in st.cfg["routes"]]}

    return app


def main():
    cfg = load_config(os.environ.get("KEV_ROUTER_CONFIG"))
    uvicorn.run(create_app(cfg), host="0.0.0.0",
                port=int(cfg["listen_port"]), log_level="warning")


if __name__ == "__main__":
    main()
