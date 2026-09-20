#!/usr/bin/env python3
"""Benchmark kev-router against a live kev + echo provider.

Usage: python3 scripts/benchmark.py [router_url]
Requires: kev server, kev-router (:8323) and any OpenAI-compatible target
(the tests/echo_provider.py works fine as target).
"""
import json
import sys
import time
import urllib.request
import urllib.error

ROUTER = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8323"
ECHO = "http://127.0.0.1:8099"

# (label, question, expected_route)
PROBES = [
    ("lookup", "What is the capital of Australia? Answer with the city name only.", "fast"),
    ("haiku", "Write a short haiku about autumn rain.", "fast"),
    ("extract", "From this sentence extract the invoice number, reply with the number only: 'Invoice INV-99123 was paid on March 3rd.'", "fast"),
    ("translate", "Translate to German: 'The router picks the smallest model that can do the job.'", "fast"),
    ("debug", "Debug this stack trace: AttributeError: 'NoneType' object has no attribute 'send' inside my HTTP client retry loop.", "code"),
    ("refactor", "Refactor this 200-line Python function into smaller units and explain the refactoring pattern you applied.", "code"),
    ("sql", "Write a SQL query joining orders and customers to find the top 5 customers by revenue last quarter, optimized for a 10M row table.", "code"),
    ("strategy", "We have 3 months of runway: cut burn 40% now or raise a down round? Analyze second-order effects of each path.", "powerful"),
    ("longdoc", "Summarize this 60-page lease agreement and flag clauses creating unlimited liability or auto-renewal traps.", "powerful"),
    ("policy", "Draft a fair usage policy balancing privacy and abuse prevention for a small AI product, with tradeoff analysis.", "powerful"),
]


def post(url, payload, timeout=60):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hdr = r.headers.get("x-kev-route", "{}")
            return json.loads(hdr or "{}"), time.time() - t0, None
    except urllib.error.HTTPError as e:
        hdr = e.headers.get("x-kev-route", "{}") if e.headers else "{}"
        return json.loads(hdr or "{}"), time.time() - t0, f"HTTP {e.code}"


def main():
    # direct baseline (no router)
    _, t_direct, _ = post(ECHO + "/v1/chat/completions",
                          {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    print(f"baseline direct->provider: {t_direct*1000:.0f} ms")
    print()
    print("| probe | expected | routed | conf | complexity | first call | cached |")
    print("|---|---|---|---|---|---|---|")
    ok = t_first = t_cached = 0
    n = 0
    for label, q, expected in PROBES:
        payload = {"model": "ignored",
                   "messages": [{"role": "user", "content": q}]}
        route, t1, err = post(ROUTER + "/v1/chat/completions", payload)
        _, t2, _ = post(ROUTER + "/v1/chat/completions", payload)  # same conv -> cache
        src = route.get("source", "?")
        r = route.get("route", "?")
        conf = route.get("confidence", 0)
        cx = route.get("complexity", 0)
        hit = "y" if r == expected else "**NO**"
        print(f"| {label} | {expected} | {r} ({src}) | {conf:.2f} | {cx} | "
              f"{t1:.2f}s{' ' + err if err else ''} | {t2*1000:.0f} ms |")
        ok += (r == expected)
        t_first += t1
        t_cached += t2
        n += 1
    print(f"\naccuracy: {ok}/{n}   avg first call: {t_first/n:.2f} s   "
          f"avg cached: {t_cached/n*1000:.0f} ms   "
          f"routing overhead (first): {(t_first/n-t_direct):.2f} s, "
          f"(cached): {(t_cached/n-t_direct)*1000:.0f} ms")


if __name__ == "__main__":
    main()
