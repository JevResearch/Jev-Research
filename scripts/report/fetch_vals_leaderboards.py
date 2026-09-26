#!/usr/bin/env python3
"""Fetch the vals.ai leaderboards and publish the canonical extract.

Writes docs/modern-comparison/canonical/vals-leaderboards-20260926.json:
for each of mmlu_pro / gpqa / math500 / hle (hlab), every model row from the
page's SSR payload (accuracy, stderr, cost_per_test, latency, reasoning and
compute effort, provider), plus curated display metadata per slug:
  name        display name
  released    release month "YYYY-MM"
  date_basis  "slug" (date embedded in the slug) | "documented" (repo/press)
              | "estimated" (family ordering)
  tier        reasoning tier for the chart symbols:
              none / low / medium / high / xhigh / max

The tier comes from the row's own config where present (reasoning_effort,
compute_effort, -thinking / non-reasoning slug tokens), else from the
family default. Costs are the vals platform's measured cost_per_test, which
already includes any chain-of-thought the model burns.

Network required. Run:  python scripts/report/fetch_vals_leaderboards.py
"""

from __future__ import annotations

import html as _html
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/canonical/vals-leaderboards-20260926.json"
PAGES = {"mmlu_pro": "mmlu_pro", "gpqa": "gpqa", "math500": "math500",
         "hle": "hlab"}
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return _html.unescape(r.read().decode("utf-8", "replace"))


def extract_overall(h: str) -> tuple[dict, str | None]:
    occ = [m.start() for m in re.finditer(r'"overall":\[0,\{', h)]
    block = None
    for i in occ:
        start = h.index("{", i + len('"overall":[0,'))
        depth, j = 0, start
        while j < len(h):
            if h[j] == "{":
                depth += 1
            elif h[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        cand = h[start:j + 1]
        if '"provider"' in cand or '"accuracy"' in cand:
            if block is None or '"provider"' in cand:
                block = cand
            if '"provider"' in cand:
                break
    models = {}
    for m in re.finditer(r'"([a-z0-9_.\-]+/[a-zA-Z0-9_.\-]+)":\[0,\{(.*?)\}(?:\]|,")',
                         block or ""):
        slug, body = m.group(1), m.group(2)

        def g(field, cast=float):
            mm = re.search(r'"%s":\[0,([^\],]+)\]' % re.escape(field), body)
            if not mm:
                return None
            v = mm.group(1).strip()
            if v == "null":
                return None
            if v.startswith('"'):
                return v.strip('"')
            try:
                return cast(v)
            except ValueError:
                return None

        models[slug] = {
            "accuracy": g("accuracy"), "stderr": g("stderr"),
            "cost_per_test": g("cost_per_test"), "latency": g("latency"),
            "reasoning_effort": g("reasoning_effort", str),
            "compute_effort": g("compute_effort", str),
            "provider": g("provider", str),
        }
    upd = re.search(r'"updated":\[0,"([0-9-]+)"\]', h)
    return models, (upd.group(1) if upd else None)


# --------------------------------------------------------------- curated meta
# Explicit release months. "slug" = embedded in the slug; "documented" = repo
# artifacts (costs.json price-sheet dates, research.md announcements);
# everything else is family-ordered estimation and labeled as such downstream.
DATES: dict[str, str] = {
    "openai/gpt-3.5-turbo": "2022-11",
    "openai/gpt-4o-2024-05-13": "2024-05",
    "openai/gpt-4o-mini-2024-07-18": "2024-07",
    "openai/gpt-4o-2024-08-06": "2024-08",
    "openai/gpt-4o-2024-11-20": "2024-11",
    "openai/o1-2024-12-17": "2024-12",
    "openai/o3-mini-2025-01-31": "2025-01",
    "openai/gpt-4.1-2025-04-14": "2025-04",
    "openai/gpt-4.1-mini-2025-04-14": "2025-04",
    "openai/gpt-4.1-nano-2025-04-14": "2025-04",
    "openai/o3-2025-04-16": "2025-04",
    "openai/o4-mini-2025-04-16": "2025-04",
    "openai/gpt-5-2025-08-07": "2025-08",
    "openai/gpt-5-mini-2025-08-07": "2025-08",
    "openai/gpt-5-nano-2025-08-07": "2025-08",
    "openai/gpt-5.1-2025-11-13": "2025-11",
    "openai/gpt-5.2-2025-12-11": "2025-12",
    "openai/gpt-5.4-2026-03-05": "2026-03",
    "openai/gpt-5.4-mini-2026-03-17": "2026-03",
    "openai/gpt-5.4-nano-2026-03-17": "2026-03",
    "openai/gpt-5.5": "2026-05",
    "openai/gpt-5.6-luna": "2026-07",
    "openai/gpt-5.6-terra": "2026-07",
    "openai/gpt-5.6-sol": "2026-07",
    "openai/gpt-6-luna": "2026-09",
    "openai/gpt-6-sol": "2026-09",
    "openai/gpt-6-astra": "2026-09",
    "anthropic/claude-3-5-haiku-20241022": "2024-10",
    "anthropic/claude-3-5-sonnet-20241022": "2024-10",
    "anthropic/claude-3-7-sonnet-20250219": "2025-02",
    "anthropic/claude-3-7-sonnet-20250219-thinking": "2025-02",
    "anthropic/claude-opus-4-20250514": "2025-05",
    "anthropic/claude-sonnet-4-20250514": "2025-05",
    "anthropic/claude-sonnet-4-20250514-thinking": "2025-05",
    "anthropic/claude-opus-4-1-20250805": "2025-08",
    "anthropic/claude-opus-4-1-20250805-thinking": "2025-08",
    "anthropic/claude-sonnet-4-5-20250929-thinking": "2025-09",
    "anthropic/claude-haiku-4-5-20251001-thinking": "2025-10",
    "anthropic/claude-opus-4-5-20251101": "2025-11",
    "anthropic/claude-opus-4-5-20251101-thinking": "2025-11",
    "anthropic/claude-opus-4-6-thinking": "2026-01",
    "anthropic/claude-sonnet-4-6": "2026-02",
    "anthropic/claude-opus-4-7": "2026-03",
    "anthropic/claude-opus-4-8": "2026-06",
    "anthropic/claude-fable-5": "2026-07",
    "anthropic/claude-fable-5-1": "2026-08",
    "anthropic/claude-opus-5": "2026-08",
    "anthropic/claude-sonnet-5": "2026-06",
    "anthropic/claude-opus-5-5": "2026-09",
    "google/gemini-1.5-flash-002": "2024-09",
    "google/gemini-1.5-pro-002": "2024-09",
    "google/gemini-2.0-flash-exp": "2024-12",
    "google/gemini-2.0-flash-001": "2025-02",
    "google/gemini-2.0-flash-thinking-exp-01-21": "2025-01",
    "google/gemini-2.5-pro-exp-03-25": "2025-03",
    "google/gemini-2.5-flash-preview-04-17": "2025-04",
    "google/gemini-2.5-flash-preview-04-17-thinking": "2025-04",
    "google/gemini-2.5-flash-preview-09-2025": "2025-09",
    "google/gemini-2.5-flash-preview-09-2025-thinking": "2025-09",
    "google/gemini-2.5-flash-lite-preview-09-2025": "2025-09",
    "google/gemini-2.5-flash-lite-preview-09-2025-thinking": "2025-09",
    "google/gemini-3-pro-preview": "2025-11",
    "google/gemini-3-flash-preview": "2025-12",
    "google/gemini-3.1-pro-preview": "2026-02",
    "google/gemini-3.1-flash-lite-preview": "2026-02",
    "google/gemini-3.5-flash": "2026-04",
    "google/gemini-3.5-flash-lite": "2026-04",
    "google/gemini-3.6-flash": "2026-06",
    "google/gemini-3.7-flash": "2026-07",
    "google/gemini-3.8-flash": "2026-08",
    "grok/grok-2-1212": "2024-12",
    "grok/grok-3": "2025-02",
    "grok/grok-3-mini-fast-low-reasoning": "2025-02",
    "grok/grok-3-mini-fast-high-reasoning": "2025-02",
    "grok/grok-4-0709": "2025-07",
    "grok/grok-4-fast-non-reasoning": "2025-09",
    "grok/grok-4-fast-reasoning": "2025-09",
    "grok/grok-4-1-fast-non-reasoning": "2025-11",
    "grok/grok-4-1-fast-reasoning": "2025-11",
    "grok/grok-4.20-0309-reasoning": "2026-03",
    "grok/grok-4.3": "2026-01",
    "grok/grok-4.5": "2026-03",
    "grok/grok-4.6": "2026-05",
    "grok/grok-4.7": "2026-08",
    "alibaba/qwen3-max": "2025-09",
    "alibaba/qwen3-max-preview": "2025-08",
    "alibaba/qwen3-max-2026-01-23": "2026-01",
    "alibaba/qwen3.5-flash": "2026-02",
    "alibaba/qwen3.5-plus-thinking": "2026-02",
    "alibaba/qwen3.6-plus": "2026-04",
    "alibaba/qwen3.7-max": "2026-06",
    "alibaba/qwen3.7-plus": "2026-06",
    "alibaba/qwen3.8-27b": "2026-08",
    "alibaba/qwen3.8-max": "2026-08",
    "zai/glm-4.5": "2025-07",
    "zai/glm-4.6": "2025-09",
    "zai/glm-4.7": "2025-12",
    "zai/glm-5-thinking": "2026-02",
    "zai/glm-5.1": "2026-04",
    "zai/glm-5.2": "2026-06",
    "zai/glm-5.3": "2026-08",
    "zai/glm-5.3-flash": "2026-08",
    "deepseek/deepseek-v4-flash-0731": "2025-07",
    "deepseek/deepseek-v4-pro-0813": "2025-08",
    "deepseek/deepseek-v4-pro": "2025-08",
    "deepseek/deepseek-v4.1-flash": "2025-11",
    "fireworks/deepseek-r1": "2025-05",
    "fireworks/deepseek-v3": "2024-12",
    "fireworks/deepseek-v3-0324": "2025-03",
    "fireworks/deepseek-v3p2": "2025-08",
    "fireworks/deepseek-v3p2-thinking": "2025-08",
    "fireworks/gpt-oss-120b": "2025-08",
    "fireworks/gpt-oss-20b": "2025-08",
    "fireworks/llama4-maverick-instruct-basic": "2025-04",
    "fireworks/nemotron-lightning-3p5-30b-a3b": "2025-09",
    "fireworks/qwen3-235b-a22b": "2025-07",
    "mistralai/mistral-small-2402": "2024-02",
    "mistralai/mistral-large-2411": "2024-11",
    "mistralai/mistral-small-2503": "2025-03",
    "mistralai/mistral-medium-2505": "2025-05",
    "mistralai/magistral-medium-2509": "2025-09",
    "mistralai/magistral-small-2509": "2025-09",
    "mistralai/mistral-large-2512": "2025-12",
    "mistralai/mistral-medium-3.5": "2026-06",
    "cohere/command-r-plus": "2024-04",
    "cohere/command-a-03-2025": "2025-03",
    "ai21labs/jamba-mini-1.6": "2024-08",
    "ai21labs/jamba-large-1.6": "2024-08",
    "kimi/kimi-k2-thinking": "2025-11",
    "kimi/kimi-k2.5-thinking": "2026-01",
    "kimi/kimi-k2.6": "2026-03",
    "kimi/kimi-k3": "2026-06",
    "minimax/MiniMax-M2.1": "2025-12",
    "minimax/MiniMax-M2.5": "2026-02",
    "minimax/MiniMax-M2.7": "2026-05",
    "minimax/MiniMax-M3": "2026-08",
    "xiaomi/mimo-v2-flash": "2025-12",
    "xiaomi/mimo-v2.5": "2026-02",
    "xiaomi/mimo-v2.5-pro": "2026-02",
    "xiaomi/mimo-v2.6-flash": "2026-05",
    "xiaomi/mimo-v2.6-pro": "2026-05",
    "meta/muse_spark": "2026-01",
    "meta/muse_spark_1_1": "2026-03",
    "meta/muse_spark_1_2": "2026-05",
    "meta/muse_spark_1_3": "2026-07",
    "meta/muse_spark_1_3_max": "2026-07",
    "nvidia/nemotron-3-ultra-550b-a55b": "2026-03",
    "thinkingmachines/inkling": "2026-04",
    "thinkingmachines/inkling-small": "2026-04",
    "ant/ling-3.0-flash-2607": "2026-07",
    "ant/ling-3.0-flash-af-rc3": "2026-08",
    "tencent/hy4-preview": "2026-06",
    "inception/mercury-2.5": "2025-10",
    "poolside/laguna-xs.2": "2026-02",
    "poolside/laguna-m.1": "2026-05",
}

DISPLAY: dict[str, str] = {
    "openai/gpt-3.5-turbo": "GPT-3.5 Turbo",
    "openai/gpt-4o-2024-05-13": "GPT-4o (05-13)",
    "openai/gpt-4o-2024-08-06": "GPT-4o (08-06)",
    "openai/gpt-4o-2024-11-20": "GPT-4o (11-20)",
    "openai/gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "openai/o1-2024-12-17": "o1",
    "openai/o3-mini-2025-01-31": "o3-mini",
    "openai/o3-2025-04-16": "o3",
    "openai/o4-mini-2025-04-16": "o4-mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "openai/gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "openai/gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai/gpt-5-nano-2025-08-07": "GPT-5 nano",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/gpt-5.2-2025-12-11": "GPT-5.2",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "openai/gpt-5.6-terra": "GPT-5.6 Terra",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6-sol": "GPT-6 Sol",
    "openai/gpt-6-astra": "GPT-6 Astra",
    "anthropic/claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "anthropic/claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "anthropic/claude-3-7-sonnet-20250219-thinking": "Claude 3.7 Sonnet (thinking)",
    "anthropic/claude-opus-4-20250514": "Claude Opus 4",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-sonnet-4-20250514-thinking": "Claude Sonnet 4 (thinking)",
    "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1",
    "anthropic/claude-opus-4-1-20250805-thinking": "Claude Opus 4.1 (thinking)",
    "anthropic/claude-sonnet-4-5-20250929-thinking": "Claude Sonnet 4.5 (thinking)",
    "anthropic/claude-haiku-4-5-20251001-thinking": "Claude Haiku 4.5 (thinking)",
    "anthropic/claude-opus-4-5-20251101": "Claude Opus 4.5",
    "anthropic/claude-opus-4-5-20251101-thinking": "Claude Opus 4.5 (thinking)",
    "anthropic/claude-opus-4-6-thinking": "Claude Opus 4.6 (thinking)",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "anthropic/claude-opus-4-7": "Claude Opus 4.7",
    "anthropic/claude-opus-4-8": "Claude Opus 4.8",
    "anthropic/claude-fable-5": "Claude Fable 5",
    "anthropic/claude-fable-5-1": "Claude Fable 5.1",
    "anthropic/claude-opus-5": "Claude Opus 5",
    "anthropic/claude-opus-5-5": "Claude Opus 5.5",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "google/gemini-1.5-flash-002": "Gemini 1.5 Flash",
    "google/gemini-1.5-pro-002": "Gemini 1.5 Pro",
    "google/gemini-2.0-flash-exp": "Gemini 2.0 Flash (exp)",
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "google/gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
    "google/gemini-2.5-pro-exp-03-25": "Gemini 2.5 Pro (exp)",
    "google/gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash (04-17)",
    "google/gemini-2.5-flash-preview-04-17-thinking": "Gemini 2.5 Flash Thinking (04-17)",
    "google/gemini-2.5-flash-preview-09-2025": "Gemini 2.5 Flash (09)",
    "google/gemini-2.5-flash-preview-09-2025-thinking": "Gemini 2.5 Flash Thinking (09)",
    "google/gemini-2.5-flash-lite-preview-09-2025": "Gemini 2.5 Flash-Lite",
    "google/gemini-2.5-flash-lite-preview-09-2025-thinking": "Gemini 2.5 Flash-Lite (thinking)",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "google/gemini-3.6-flash": "Gemini 3.6 Flash",
    "google/gemini-3.7-flash": "Gemini 3.7 Flash",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "grok/grok-2-1212": "Grok 2",
    "grok/grok-3": "Grok 3",
    "grok/grok-3-mini-fast-low-reasoning": "Grok 3 Mini (low)",
    "grok/grok-3-mini-fast-high-reasoning": "Grok 3 Mini (high)",
    "grok/grok-4-0709": "Grok 4",
    "grok/grok-4-fast-non-reasoning": "Grok 4 Fast",
    "grok/grok-4-fast-reasoning": "Grok 4 Fast (reasoning)",
    "grok/grok-4-1-fast-non-reasoning": "Grok 4.1 Fast",
    "grok/grok-4-1-fast-reasoning": "Grok 4.1 Fast (reasoning)",
    "grok/grok-4.20-0309-reasoning": "Grok 4.20 (reasoning)",
    "grok/grok-4.3": "Grok 4.3",
    "grok/grok-4.5": "Grok 4.5",
    "grok/grok-4.6": "Grok 4.6",
    "grok/grok-4.7": "Grok 4.7",
    "alibaba/qwen3-max": "Qwen3 Max",
    "alibaba/qwen3-max-preview": "Qwen3 Max (preview)",
    "alibaba/qwen3-max-2026-01-23": "Qwen3 Max (01-23)",
    "alibaba/qwen3.5-flash": "Qwen3.5 Flash",
    "alibaba/qwen3.5-plus-thinking": "Qwen3.5 Plus (thinking)",
    "alibaba/qwen3.6-plus": "Qwen3.6 Plus",
    "alibaba/qwen3.7-max": "Qwen3.7 Max",
    "alibaba/qwen3.7-plus": "Qwen3.7 Plus",
    "alibaba/qwen3.8-27b": "Qwen3.8-27B",
    "alibaba/qwen3.8-max": "Qwen3.8 Max",
    "zai/glm-4.5": "GLM-4.5",
    "zai/glm-4.6": "GLM-4.6",
    "zai/glm-4.7": "GLM-4.7",
    "zai/glm-5-thinking": "GLM-5 (thinking)",
    "zai/glm-5.1": "GLM-5.1",
    "zai/glm-5.2": "GLM-5.2",
    "zai/glm-5.3": "GLM-5.3",
    "zai/glm-5.3-flash": "GLM-5.3 Flash",
    "deepseek/deepseek-v4-flash-0731": "DeepSeek V4 Flash 0731",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro 0813",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "fireworks/deepseek-r1": "DeepSeek R1 (0528)",
    "fireworks/deepseek-v3": "DeepSeek V3",
    "fireworks/deepseek-v3-0324": "DeepSeek V3 (0324)",
    "fireworks/deepseek-v3p2": "DeepSeek V3.2",
    "fireworks/deepseek-v3p2-thinking": "DeepSeek V3.2 (thinking)",
    "fireworks/gpt-oss-120b": "gpt-oss-120b",
    "fireworks/gpt-oss-20b": "gpt-oss-20b",
    "fireworks/llama4-maverick-instruct-basic": "Llama 4 Maverick",
    "fireworks/nemotron-lightning-3p5-30b-a3b": "Nemotron 3.5 30B-A3B",
    "fireworks/qwen3-235b-a22b": "Qwen3-235B-A22B",
    "mistralai/mistral-small-2402": "Mistral Small (2402)",
    "mistralai/mistral-large-2411": "Mistral Large 2 (2411)",
    "mistralai/mistral-small-2503": "Mistral Small (2503)",
    "mistralai/mistral-medium-2505": "Mistral Medium (2505)",
    "mistralai/magistral-medium-2509": "Magistral Medium",
    "mistralai/magistral-small-2509": "Magistral Small",
    "mistralai/mistral-large-2512": "Mistral Large (2512)",
    "mistralai/mistral-medium-3.5": "Mistral Medium 3.5",
    "cohere/command-r-plus": "Command R+",
    "cohere/command-a-03-2025": "Command A",
    "ai21labs/jamba-mini-1.6": "Jamba Mini 1.6",
    "ai21labs/jamba-large-1.6": "Jamba Large 1.6",
    "kimi/kimi-k2-thinking": "Kimi K2 Thinking",
    "kimi/kimi-k2.5-thinking": "Kimi K2.5 Thinking",
    "kimi/kimi-k2.6": "Kimi K2.6",
    "kimi/kimi-k3": "Kimi K3",
    "minimax/MiniMax-M2.1": "MiniMax M2.1",
    "minimax/MiniMax-M2.5": "MiniMax M2.5",
    "minimax/MiniMax-M2.7": "MiniMax M2.7",
    "minimax/MiniMax-M3": "MiniMax M3",
    "xiaomi/mimo-v2-flash": "MiMo v2 Flash",
    "xiaomi/mimo-v2.5": "MiMo v2.5",
    "xiaomi/mimo-v2.5-pro": "MiMo v2.5 Pro",
    "xiaomi/mimo-v2.6-flash": "MiMo v2.6 Flash",
    "xiaomi/mimo-v2.6-pro": "MiMo v2.6 Pro",
    "meta/muse_spark": "Muse Spark",
    "meta/muse_spark_1_1": "Muse Spark 1.1",
    "meta/muse_spark_1_2": "Muse Spark 1.2",
    "meta/muse_spark_1_3": "Muse Spark 1.3",
    "meta/muse_spark_1_3_max": "Muse Spark 1.3 Max",
    "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B-A55B",
    "thinkingmachines/inkling": "Inkling",
    "thinkingmachines/inkling-small": "Inkling Small",
    "ant/ling-3.0-flash-2607": "Ling 3.0 Flash",
    "ant/ling-3.0-flash-af-rc3": "Ling 3.0 Flash (rc3)",
    "tencent/hy4-preview": "Hunyuan 4 (preview)",
    "inception/mercury-2.5": "Mercury 2.5",
    "poolside/laguna-xs.2": "Laguna XS.2",
    "poolside/laguna-m.1": "Laguna M.1",
}

# family default reasoning tiers where the row config is silent
TIER_DEFAULTS = {
    "o1": "high", "o3": "high", "o4": "high",
    "gpt-5": "high", "gpt-6": "high",
    "gemini-2.0-flash-thinking": "medium", "gemini-2.5": "medium",
    "gemini-3": "high", "magistral": "high", "deepseek-r1": "high",
    "kimi-k2-thinking": "high", "kimi-k2.5": "high", "kimi-k3": "high",
    "glm-5": "high", "muse_spark_1_3_max": "max", "muse_spark": "high",
    "nemotron-3-ultra": "high", "inkling": "medium",
    "claude-fable": "high", "claude-opus-5": "high", "claude-sonnet-5": "high",
}


def date_basis(slug: str) -> str:
    if re.search(r"20\d{6}|20\d{2}-\d{2}|\d{4}$|2[456]\d{2}", slug):
        return "slug"
    return "documented" if slug in DATES else "estimated"


def tier_of(slug: str, row: dict) -> str:
    ce = (row.get("compute_effort") or "").lower()
    re_ = (row.get("reasoning_effort") or "").lower()
    if ce == "max":
        return "max"
    if re_ in ("max", "xhigh"):
        return "xhigh"
    if re_ == "high":
        return "high"
    if re_ == "medium":
        return "medium"
    if re_ == "low":
        return "low"
    if re_ == "none":
        return "none"
    if "non-reasoning" in slug:
        return "none"
    if "-thinking" in slug or "reasoning" in slug:
        return "high"
    for key, t in TIER_DEFAULTS.items():
        if key in slug:
            return t
    return "none"


def main() -> int:
    out: dict = {
        "retrieved": "2026-09-26",
        "source": "vals.ai benchmark pages (SSR payload), fetched by "
                  "scripts/report/fetch_vals_leaderboards.py",
        "protocol_note": "vals platform harness; per-row config preserved "
                         "(reasoning_effort, compute_effort). cost_per_test is "
                         "the platform's measured cost per test including any "
                         "chain-of-thought tokens.",
        "meta_caveat": "release months marked 'estimated' are family-ordered "
                       "estimates used ONLY for the era color gradient; they "
                       "are not publication claims.",
        "benchmarks": {},
    }
    for key, page in PAGES.items():
        h = fetch(f"https://www.vals.ai/benchmarks/{page}")
        models, updated = extract_overall(h)
        rows = {}
        for slug, row in models.items():
            if row.get("accuracy") is None:
                continue
            rows[slug] = dict(row)
            rows[slug]["name"] = DISPLAY.get(slug, slug.split("/")[-1])
            rows[slug]["released"] = DATES.get(slug)
            rows[slug]["date_basis"] = date_basis(slug)
            rows[slug]["tier"] = tier_of(slug, row)
        out["benchmarks"][key] = {"page_updated": updated, "n": len(rows),
                                  "models": rows}
        print(f"{key}: {len(rows)} models (page updated {updated})")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
