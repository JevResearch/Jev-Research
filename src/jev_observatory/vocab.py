"""English vocabulary with frequencies for word-completion decoding.

A "reasonable subset" of English (zipf >= 2.3, i.e. common words only) plus
task-relevant additions (this project's names, AI vocabulary, common company
and product names) at moderate frequencies.  Cached as JSON so runs are
reproducible; the cache records its construction parameters.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

TASK_WORDS: dict[str, float] = {
    # this project
    "jev": 3.2, "typesafe": 3.0, "mellum": 3.0, "jetbrains": 3.2,
    # AI / assistant vocabulary
    "ai": 3.6, "llm": 3.2, "chatbot": 3.2, "chatbots": 3.0, "prompt": 3.2,
    "prompts": 3.0, "inference": 3.2, "tokenizer": 2.8, "chatgpt": 3.6,
    "openai": 3.6, "anthropic": 3.2, "claude": 3.2, "gemini": 3.4,
    "copilot": 3.4, "deepseek": 3.0, "mistral": 3.0, "huggingface": 2.8,
    "deepmind": 3.0, "nvidia": 3.6, "microsoft": 3.8, "google": 4.4,
    "meta": 3.6, "llama": 3.4, "grok": 3.2, "alexa": 3.2, "siri": 3.2,
}

MIN_ZIPF = 2.3
MAX_WORD_LEN = 15  # partial words longer than this are forced to end
_CACHE_PATH = Path(__file__).resolve().parents[2] / "models" / "word_vocab.json"


class WordVocab:
    """word -> zipf frequency, plus prefix-index helpers."""

    def __init__(self, words: dict[str, float]) -> None:
        self.freq = dict(words)
        self._children: dict[str, list[tuple[str, float]]] = {}
        for word, f in self.freq.items():
            for i in range(1, len(word) + 1):
                self._children.setdefault(word[:i], []).append((word, f))

    # ------------------------------------------------------------- lookups
    def mass(self, prefix: str) -> float:
        """Total frequency of vocabulary words starting with `prefix`."""
        return sum(f for _, f in self._children.get(prefix, ()))

    def completions(self, prefix: str) -> list[tuple[str, float]]:
        """Vocabulary words starting with `prefix` (descending frequency)."""
        return sorted(self._children.get(prefix, ()), key=lambda wf: -wf[1])

    def complete_word(self, prefix: str) -> bool:
        return prefix in self.freq

    def top_words(self, n: int, *, exclude: frozenset[str] = frozenset()) -> list[tuple[str, float]]:
        pool = [(w, f) for w, f in self.freq.items() if w not in exclude]
        return sorted(pool, key=lambda wf: -wf[1])[:n]


def build_vocab() -> WordVocab:
    """Load from cache if valid, else build from wordfreq and cache."""
    if _CACHE_PATH.exists():
        doc = json.loads(_CACHE_PATH.read_text())
        if doc.get("min_zipf") == MIN_ZIPF and doc.get("task_words") == TASK_WORDS:
            return WordVocab(doc["words"])
    from wordfreq import zipf_frequency

    words = {w: zipf_frequency(w, "en") for w in _candidate_pool()}
    words = {w: f for w, f in words.items() if f >= MIN_ZIPF}
    for w, f in TASK_WORDS.items():
        words.setdefault(w, f)
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    _CACHE_PATH.write_text(json.dumps({
        "min_zipf": MIN_ZIPF, "task_words": TASK_WORDS, "words": words,
    }))
    return WordVocab(words)


def _candidate_pool() -> list[str]:
    from wordfreq import iter_wordlist

    return list(iter_wordlist("en"))
