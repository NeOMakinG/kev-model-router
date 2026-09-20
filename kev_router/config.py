"""Config loading for kev-router.

Routes are declared in a YAML (or JSON) file. Secrets (API keys) are NEVER
stored in config files -- providers are referenced by env var name, and the
key itself is read from the environment at request time.
"""
import json
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # yaml is optional; JSON config works without it
    yaml = None

DEFAULT_CONFIG: dict = {
    "kev_url": "http://127.0.0.1:8009/v1/systemone",
    "listen_port": 8323,
    "default_route": "fast",
    "kev_timeout_s": 4.0,
    "cache_ttl_s": 3600,
    "max_state_chars": 6000,
    "complexity_threshold": 3.0,
    "routes": {
        "fast": {
            "target": {"type": "env", "env": "KEV_ROUTER_TARGET_FAST"},
            "criteria": "Direct lookups, extraction, classification, short localized edits",
        },
        "code": {
            "target": {"type": "env", "env": "KEV_ROUTER_TARGET_CODE"},
            "criteria": "Code, debugging, stack traces, refactoring, technical implementation",
        },
        "powerful": {
            "target": {"type": "env", "env": "KEV_ROUTER_TARGET_POWERFUL"},
            "criteria": "Complex reasoning, architecture, long documents, high-stakes decisions",
        },
    },
    "complexity_route": "powerful",
}

_env_expand = os.path.expandvars


def _deep_expand(value):
    if isinstance(value, str):
        return _env_expand(value)
    if isinstance(value, dict):
        return {k: _deep_expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_expand(v) for v in value]
    return value


def _resolve_target(t):
    if not isinstance(t, dict):
        return str(t)
    if t.get("type") == "env":
        val = os.environ.get(t.get("env", ""), "")
        if val:
            return val.rstrip("/")
        return (t.get("fallback") or "").rstrip("/") or None
    return str(t.get("url", "")).rstrip("/") or None


def load_config(path: str | None = None) -> dict:
    """Load config from YAML/JSON file, env-overridden, with sane defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    candidates = ([path] if path else
                  [os.environ.get("KEV_ROUTER_CONFIG", ""),
                   "kev-router.yaml", "kev-router.json",
                   str(Path.home() / ".config" / "kev-router" / "config.yaml")])
    for cand in candidates:
        if cand and os.path.exists(cand):
            raw = Path(cand).read_text()
            data = (yaml.safe_load(raw) if yaml and cand.endswith((".yaml", ".yml"))
                    else json.loads(raw))
            cfg.update({k: v for k, v in (data or {}).items() if v is not None})
            break

    cfg = _deep_expand(cfg)

    # Resolve provider targets: {"type": "env", ...} -> concrete base URL
    resolved = {}
    for name, route in cfg.get("routes", {}).items():
        base = _resolve_target(route.get("target", {}))
        resolved[name] = {**route, "base_url": base}
    cfg["routes"] = resolved

    # kev_api_key: reference by env var NAME, resolved at request time.
    kev_cfg = cfg.get("kev", {})
    if isinstance(kev_cfg, dict) and kev_cfg.get("api_key_env"):
        cfg["kev_api_key_env"] = kev_cfg["api_key_env"]
    return cfg


def get_kev_api_key(cfg: dict) -> str | None:
    env_name = cfg.get("kev_api_key_env")
    return os.environ.get(env_name) if env_name else None
