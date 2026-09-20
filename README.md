# kev-router

**Jev-style model routing, powered by [kev](https://github.com/jaredpalmer/kev) — 100% local, 100% free.**

Route every request to the right model with a typed decision model instead of
keyword rules or a paid LLM judge. kev (a System One decision model: a LoRA +
readout head over Qwen) classifies each request in a single forward pass —
**~0.15 s** with kev-0.6b, **~1.2 s** with kev-4b on a CPU — and the request is
forwarded to the provider that route points at.

The same pattern as the Jev router plugins for Claude Code / LangChain /
OpenCode, but the decision brain is local and costs nothing.

## How it works

```
 your agent (Claude Code, Hermes, any OpenAI client)
        │  POST /v1/chat/completions
        ▼
 ┌─────────────────────┐   state = last user message
 │    kev-router       │──────────────► kev /v1/systemone
 │  (OpenAI-compatible)│   ◄── {route: "powerful", confidence: 0.91,
 │                     │        complexity: 2.7}          (~0.1-1.5 s)
 └─────────┬───────────┘
           │ model rewritten, auth headers PASSED THROUGH untouched
           ▼
   http://your-provider/v1/chat/completions   (OpenRouter, cloud API, llama.cpp…)
```

Design rules (inherited from the Jev ecosystem):

1. **Fail-open** — kev unreachable, timeout, bad answer → default route. The
   router never blocks traffic.
2. **Per-conversation decision cache** — the route is decided once per
   conversation (key = first user message), like the Jev Claude Code mod only
   routes the main model at session start: switching models mid-conversation
   would invalidate the provider's prompt cache.
3. **Complexity override** — if kev says `fast` but scores complexity ≥ 3
   (complex), the request is upgraded to the `complexity_route`.
4. **Zero secrets stored** — provider URLs come from env vars declared in the
   config; API keys are passed through from the client untouched. The router
   itself holds no credentials. (If your kev endpoint itself needs a bearer,
   reference its *env var name* — never the key.)
5. **One log line per decision.**

## Quick start

```bash
# 1. run kev (see https://github.com/jaredpalmer/kev)
git clone https://github.com/jaredpalmer/kev && cd kev
uv sync --extra serve
.venv/bin/python -m kev.serve --run jaredpalmer/kev-4b --port 8009

# 2. run kev-router
pip install -e "/path/to/kev-router[yaml]"
export KEV_ROUTER_TARGET_FAST="http://localhost:11434/v1"        # e.g. Ollama
export KEV_ROUTER_TARGET_POWERFUL="https://openrouter.ai/api/v1" # e.g. OpenRouter
kev-router   # listens on :8323

# 3. point your agent at the router (normal OpenAI client, any key works
#    since the router forwards auth untouched)
curl http://localhost:8323/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"ignored","messages":[{"role":"user","content":"Refactor this class..."}]}'
```

The response carries an `x-kev-route` header showing the decision:

```
x-kev-route: {"route":"powerful","confidence":0.91,"complexity":2.8,"source":"kev"}
```

## Configuration

Copy `kev-router.example.yaml` → `kev-router.yaml` (or
`~/.config/kev-router/config.yaml`). Minimal version:

```yaml
kev_url: http://127.0.0.1:8009/v1/systemone
default_model: fast
complexity_route: powerful
routes:
  fast:
    target: {type: env, env: KEV_ROUTER_TARGET_FAST}
    model: qwen3:0.6b
    criteria: "Direct lookups, extraction, short localized edits"
  powerful:
    target: {type: env, env: KEV_ROUTER_TARGET_POWERFUL}
    model: anthropic/claude-opus-4.6
    criteria: "Complex reasoning, long documents, high-stakes decisions"
```

Any number of routes works — add `coding`, `cheap`, `vision`… kev picks among
the criteria you declare. `target: {type: env, env: VARNAME}` keeps URLs out of
the file; `model` is what gets written into the forwarded request body
(omit it to pass the client's model name through unchanged).

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible; routed |
| `POST /v1/messages` | Anthropic-compatible (for Claude Code); routed |
| `GET /v1/models` | Lists your route names as model ids |
| `GET /router/health` | Stats: routed / cache hits / fail-opens / upgrades |

## Limits (honest ones)

- kev context is ~8k tokens: the router classifies on the **last user message
  (first 6000 chars)**, not the whole conversation.
- kev-4b is ~1.2 s/request on CPU (kev-0.6b: ~0.15 s). With the conversation
  cache that's once per conversation, not per request.
- The router forwards `Authorization`/`x-api-key` to **all** routes: each of
  your providers must accept the same key (env-var-per-provider keys are on
  the roadmap).

## License

MIT

## Benchmark (live, kev-4b on CPU, 3 routes: fast / code / powerful)

10 probe questions, one pass each, same conversation caching measured per probe:

| probe | expected | routed | confidence | first call | cached |
|---|---|---|---|---|---|
| lookup | fast | fast | 0.99 | 1.05 s | 5 ms |
| haiku | fast | fast | 0.97 | 1.09 s | 5 ms |
| extract | fast | fast | 1.00 | 1.32 s | 6 ms |
| translate | fast | fast | 0.43 | 1.19 s | 5 ms |
| debug stack trace | code | code | 0.92 | 1.23 s | 7 ms |
| refactor | code | code | 0.94 | 1.24 s | 5 ms |
| sql | code | code | 0.93 | 1.27 s | 5 ms |
| strategy | powerful | powerful | 0.99 | 1.26 s | 6 ms |
| longdoc | powerful | powerful | 0.80 | 1.23 s | 6 ms |
| policy | powerful | powerful | 0.95 | 1.22 s | 6 ms |

**Accuracy 10/10.** Routing overhead: ~1.2 s once per conversation (kev-4b on
CPU), then ~0 ms thanks to the per-conversation cache. Run it yourself:
`python3 scripts/benchmark.py` (needs a kev server and the echo provider).
