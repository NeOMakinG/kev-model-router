"""kev-router server: OpenAI-compatible proxy that routes with kev.

Every incoming chat request is classified by kev (one forward pass, no text
generation) which picks the route; the request is then forwarded with its
original auth headers to the provider behind that route. kev being down never
blocks traffic (fail-open to the default route).
"""
import hashlib
import json
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_kev_api_key, load_config

PASS_HEADERS = ("authorization", "x-api-key", "content-type", "accept",
                "anthropic-version", "x-request-id", "user-agent")


class RouterState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cache: dict[str, tuple[dict, float]] = {}
        self.stats = {"routed": 0, "cache_hits": 0, "fail_open": 0,
                      "upgraded": 0, "errors": 0}

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
            c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
        return str(c)[: int(os.environ.get("KEV_ROUTER_MAX_STATE_CHARS",
                                           str(6000)))]

    def decide(self, messages: list) -> dict:
        key = self.conv_key(messages)
        hit = self.cache.get(key)
        now = time.time()
        if hit and now - hit[1] < self.cfg["cache_ttl_s"]:
            self.stats["cache_hits"] += 1
            return {**hit[0], "source": "cache"}

        route_name = None
        complexity = None
        conf = 0.0
        ans = self.ask_kev(self.last_user(messages))
        if ans:
            rc = ans.get("model_route", {})
            name = rc.get("choice")
            if name in self.cfg["routes"]:
                route_name, conf = name, rc.get("confidence", 0.0)
            complexity = ans.get("complexity", {}).get("score")

        if route_name is None:  # fail-open
            self.stats["fail_open"] += 1
            route_name = self.cfg["default_model"]
            source = "fail-open"
        else:
            self.stats["routed"] += 1
            source = "kev"
            # complexity override: cheap route but heavy task -> upgrade
            if (route_name != self.cfg["complexity_route"]
                    and complexity is not None
                    and complexity >= self.cfg["complexity_threshold"]
                    and self.cfg["complexity_route"] in self.cfg["routes"]):
                route_name = self.cfg["complexity_route"]
                self.stats["upgraded"] += 1

        route = self.cfg["routes"][route_name]
        decision = {"route": route_name, "target": route.get("base_url"),
                    "model": route.get("model"), "confidence": round(conf, 2),
                    "complexity": complexity, "source": source}
        self.cache[key] = (decision, now)
        return decision


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
                        with c.stream("POST", f"{target}{path}", json=body,
                                      headers=fwd) as r:
                            for chunk in r.iter_raw():
                                yield chunk
                return StreamingResponse(gen(), media_type="text/event-stream",
                                         headers={"x-kev-route": json.dumps(dec)})
            async with httpx.AsyncClient(timeout=600) as c:
                r = await c.post(f"{target}{path}", json=body, headers=fwd)
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
