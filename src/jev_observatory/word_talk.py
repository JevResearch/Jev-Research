"""Word-completion Talk: Jev writes with a vocabulary-backed option menu.

Design (user-specified): the option set at each step is built from English
statistics, not from a local language model's logits:

* fixed menu: all letters (as extensions of the current partial word) and a
  small spelled-out punctuation set, including ``[backspace]`` so Jev can
  edit, and ``[end (finish the response)]``;
* multiletter combos ranked by *how many words share the extension, weighted
  by how common they are* (prefix mass from a frequency vocabulary);
* common words offered whole; completed words get next-word-start options
  (``helicopter a``, ``helicopter an``);
* **stuck escape**: if nothing completes the current partial word (e.g.
  ``jzqg``), the options become ``jzqg a``, ``jzqg b``, ... -- Jev sees the
  garbage it typed and can endorse abandoning it, or keep typing letters;
* **context split**: the state carries only the committed text up to the last
  word boundary; the partial word lives inside the option texts (``my name
  is `` + options ``ja``, ``jb``, ...), so Jev reads extensions in context.

Honesty: local transforms are lowercase-only emission with recorded
sentence-start capitalization; stops remain explicit; the raw distribution is
stored per step.  This is deliberately as "free form" as the menu allows:
Jev can spell any word letter-by-letter, coin words, or commit garbage
visibly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .manifest import utc_now
from .providers import BaseProvider, NullHook
from .schema import SystemOneRequest
from .token_talk import (
    END_TOKEN,
    DECODER_MODES,
    DecoderConfig,
    TokenTalkLimits,
    creates_adjacent_repeat,
)
from .vocab import MAX_WORD_LEN, WordVocab

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
PUNCT = (
    ("[space (end-of-word)]", " "),
    ("[period . (end-of-sentence)]", ". "),
    ("[comma , (pause)]", ", "),
    ("[question mark ? (ask)]", "? "),
    ("[newline]", "\n"),
    ("[backspace (remove last letter)]", "\b"),
)
BACKSPACE = "\b"


@dataclass(frozen=True)
class WordOption:
    text: str          # what Jev sees (in-context extension)
    emit: str          # what we append (or backspace marker)
    kind: str
    score: float
    is_punct: bool = False
    is_end: bool = False


def build_options(
    committed: str, partial: str, vocab: WordVocab, *,
    banned_words: frozenset[str] = frozenset(),
    max_total: int = 254,
) -> list[WordOption]:
    """The full menu for one step, ordered by extension-mass."""
    options: list[WordOption] = []
    stuck = bool(partial) and vocab.mass(partial) == 0.0 and not vocab.complete_word(partial)
    forced_end = len(partial) >= MAX_WORD_LEN
    prefix_tail = committed[-2:] if committed else ""

    def letter_ok(c: str) -> bool:
        # structural run ban: never offer a third identical letter in a row
        return not (len(partial) >= 2 and partial.endswith(c * 2))

    # 1. plain letter continuations (or escape continuations if stuck/forced)
    for c in ALPHABET:
        if not letter_ok(c):
            continue
        if stuck or forced_end:
            options.append(WordOption(f"{partial} {c}", f" {c}", "escape", 0.0))
            if not forced_end:
                options.append(WordOption(f"{partial}{c}", c, "letter", 0.0))
        else:
            ext = partial + c
            options.append(WordOption(ext, c, "letter", vocab.mass(ext)))
    # 2. multiletter combos (within-word, and escape/next-word two-letter starts)
    if not forced_end:
        combos: list[tuple[str, float]] = []
        for c1 in ALPHABET:
            if not letter_ok(c1):
                continue
            for c2 in ALPHABET:
                ext = partial + c1 + c2
                m = vocab.mass(ext)
                if m > 0:
                    combos.append((c1 + c2, m))
        combos.sort(key=lambda cm: -cm[1])
        for c1c2, m in combos[:16]:
            options.append(WordOption(partial + c1c2, c1c2, "combo", m))
    # 3. whole-word completions and next-word options
    completions = [(w, f) for w, f in vocab.completions(partial)
                   if w != partial and w.lower() not in banned_words][:8]
    next_bases = [w for w, _ in completions[:2]]
    if vocab.complete_word(partial) and partial.lower() not in banned_words:
        # the partial itself is a finished word: its next-word starts are the
        # first options (the zero-emit 'commit what is already typed' option
        # is deliberately NOT offered - it would be a no-op loop)
        next_bases.insert(0, partial)
    for w, f in completions:
        emit = w[len(partial):]
        if emit:
            options.append(WordOption(w, emit, "word", f))
    for base in next_bases[:2]:
        emit_prefix = base[len(partial):] + " "
        for c in ALPHABET:
            m = vocab.mass(c)
            if m > 0:
                options.append(WordOption(f"{base} {c}", f"{emit_prefix}{c}",
                                          "nextword_letter", m))
        seen = 0
        for c1 in ALPHABET:
            for c2 in ALPHABET:
                if seen >= 8:
                    break
                m = vocab.mass(c1 + c2)
                if m > 0:
                    options.append(WordOption(f"{base} {c1}{c2}",
                                              f"{emit_prefix}{c1}{c2}",
                                              "nextword_combo", m))
                    seen += 1
    # 4. at an empty partial, common whole words are suggested outright
    if not partial:
        seen = 0
        for w, f in vocab.top_words(400, exclude=banned_words | set(ALPHABET)):
            if seen >= 32:
                break
            if len(w) == 1 or not w.isalpha():
                continue
            options.append(WordOption(w, w, "word", f))
            seen += 1
    # 5. spelled-out punctuation + END (fixed slots, last: recency boost)
    for name, emit in PUNCT:
        if name.startswith("[backspace") and not (partial or committed):
            continue
        if name.startswith("[space") and not partial:
            continue  # no word to end: a bare gap is a no-op inviting space spam
        options.append(WordOption(name, emit, "punct", 0.0, is_punct=True))
    options.append(WordOption("[end (finish the response)]", END_TOKEN, "end",
                              0.0, is_end=True))

    options.sort(key=lambda o: (-o.score, o.text))
    # fixed punctuation/END keep their tail position regardless of score
    scored = [o for o in options if not o.is_punct and not o.is_end]
    fixed = [o for o in options if o.is_punct or o.is_end]
    return (scored + fixed)[:max_total]


def auto_capitalize(committed: str, emit: str) -> str:
    """Recorded local transform: sentence starts get a capital letter."""
    if not emit or not emit[0].isalpha() or not emit[0].islower():
        return emit
    prior = committed.rstrip()
    if not prior or prior[-1] in ".!?\n":
        return emit[0].upper() + emit[1:]
    return emit


class WordTalkSession:
    """Same honesty frame as TokenTalkSession, vocabulary-backed options."""

    def __init__(
        self,
        session_id: str,
        provider: BaseProvider,
        vocab: WordVocab,
        *,
        user_prompt: str,
        previous_turns: list[dict[str, str]] | None = None,
        limits: TokenTalkLimits | None = None,
        decoder: DecoderConfig | None = None,
        model: str = "jev-1.13.0",
        instructions: str | None = None,
        ensemble: int = 1,
        clock: Callable[[], float] = time.monotonic,
        hook: Any = None,
    ) -> None:
        previous_turns = previous_turns or []
        limits = limits or TokenTalkLimits()
        decoder = decoder or DecoderConfig()
        if decoder.mode not in ("jev_greedy", "jev_sample"):
            raise ValueError("word_talk supports jev_greedy / jev_sample only "
                             "(no local LM to score or multiply)")
        self.session_id = session_id
        self.provider = provider
        self.vocab = vocab
        self.user_prompt = user_prompt
        self.previous_turns = previous_turns
        self.limits = limits
        self.decoder = decoder
        self.model = model
        self.ensemble = max(1, ensemble)
        self.instructions = instructions or (
            "Continue the assistant response. The options show the next "
            "thing to write in context: letters extend the current word, "
            "whole words finish it, options with a space start the next "
            "word, punctuation is spelled out in brackets. Choose "
            "[end (finish the response)] when the response is complete."
        )
        self.clock = clock
        self.hook = hook or NullHook()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.current_node_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.cancelled = False
        self.started_at: float | None = None
        self._rng = __import__("random").Random(decoder.seed) if decoder.seed is not None else None
        self._provider_requests = 0
        self._noop_steps = 0

    # ------------------------------------------------------------- state
    @property
    def prefix(self) -> str:
        out = ""
        for node in self._path():
            if node["deletes"]:
                out = out[: len(out) - node["deletes"]]
            else:
                out += node["text"]
        return out

    def _path(self) -> list[dict[str, Any]]:
        path = []
        node_id = self.current_node_id
        while node_id is not None:
            path.append(self.nodes[node_id])
            node_id = self.nodes[node_id]["parent_id"]
        return list(reversed(path))

    def _committed_and_partial(self) -> tuple[str, str]:
        prefix = self.prefix
        cut = len(prefix.rstrip())
        if cut < len(prefix):  # trailing spaces: not part of a partial word
            return prefix[:cut] + prefix[cut:], ""
        cut2 = prefix.rfind(" ")
        cut2 = max(cut2, prefix.rfind("\n"))
        return prefix[: cut2 + 1], prefix[cut2 + 1:]

    def _recent_words(self, depth: int = 4) -> frozenset[str]:
        """Words actually on the page (last `depth`), lowercased: the ban must
        track the *text*, not per-node deltas (an 'an' completed as 'n' would
        otherwise never be banned)."""
        words = [w.lower() for w in self.prefix.split() if w]
        return frozenset(words[-depth:])

    def cancel(self) -> None:
        self.cancelled = True

    # -------------------------------------------------------------- stepping
    def step(self) -> dict[str, Any]:
        if self.stop_reason is not None:
            return {"type": "stop", "reason": self.stop_reason, "already_stopped": True}
        if self.cancelled:
            return self._stop("cancelled")
        if self.started_at is None:
            self.started_at = self.clock()
        if self.clock() - self.started_at > self.limits.max_seconds:
            return self._stop("deadline")
        committed, partial = self._committed_and_partial()
        if len(committed) >= self.limits.max_tokens:
            return self._stop("max_tokens")
        if self.limits.max_provider_requests is not None and \
                self._provider_requests >= self.limits.max_provider_requests:
            return self._stop("max_provider_requests")
        if self._repetition_stop(committed):
            return self._stop("repetition")
        if self._noop_steps >= 6:
            return self._stop("no_progress")

        options = build_options(committed, partial, self.vocab,
                                banned_words=self._recent_words())
        if self.ensemble > 1:
            before = len(options)
            options = [
                o for o in options
                if o.is_end or o.emit == BACKSPACE
                or not creates_adjacent_repeat(self.prefix, o.emit)
            ]
            if len(options) != before:
                self.events.append({"type": "anti_repeat",
                                    "dropped": before - len(options)})
        else:
            options = [
                o for o in options
                if o.is_end or o.emit == BACKSPACE
                or not creates_adjacent_repeat(self.prefix, o.emit)
            ]
        if not options:
            return self._stop("no_candidates")
        self._provider_requests += self.ensemble
        try:
            raw_probs, ensemble_raw, stats = self._ask_ensemble(
                committed, options, f"word-talk:{self.session_id}:{len(self.events)}")
        except Exception as exc:
            from .guards import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                return self._stop("budget_exceeded")
            return self._stop("provider_failure", detail=f"exception:{type(exc).__name__}")
        transformed = self.decoder.transform(raw_probs)
        keys = [f"opt_{i:03d}" for i in range(len(options))] + [END_TOKEN]
        if self.decoder.mode == "jev_greedy":
            chosen_key = self._argmax(transformed, keys)
        else:
            weights = self.decoder.truncate(transformed)
            if self._rng is not None:
                chosen_key = self._rng.choices(
                    list(weights), weights=list(weights.values()), k=1)[0]
            else:
                chosen_key = self._argmax(weights, keys)

        index = None if chosen_key == END_TOKEN else int(chosen_key[4:])
        option = options[index] if index is not None else None
        deletes = 1 if (option is not None and option.emit == BACKSPACE) else 0
        emit = "" if deletes else (option.emit if option is not None else "")
        cased = False
        if emit and emit[0].isalpha():
            new_emit = auto_capitalize(self.prefix, emit)
            if new_emit != emit:
                emit, cased = new_emit, True
        node = {
            "node_id": f"n{len(self.nodes):04d}",
            "parent_id": self.current_node_id,
            "text": emit, "deletes": deletes,
            "option": None if option is None else {
                "text": option.text, "kind": option.kind, "score": option.score},
            "jev_probabilities": raw_probs,
            "transformed": transformed if transformed != raw_probs else None,
            "cased": cased,
            "ensemble_raw": ensemble_raw,
            "model_returned": stats[0],
            "latency_ms": stats[1],
            "usage_input_tokens": stats[2],
        }
        self.nodes[node["node_id"]] = node
        self.current_node_id = node["node_id"]
        self._noop_steps = self._noop_steps + 1 if (not emit and not deletes) else 0
        event = {
            "type": "step", "node_id": node["node_id"], "key": chosen_key,
            "text": emit, "deletes": deletes,
            "option_text": option.text if option else END_TOKEN,
            "option_kind": option.kind if option else "end",
            "jev_top": self._top(transformed, 5),
            "model_returned": node["model_returned"], "latency_ms": node["latency_ms"],
            "ensemble_raw": node.get("ensemble_raw"),
        }
        self.events.append(event)
        if option is None or option.is_end:
            return self._stop("end_token")
        return event

    def _ask_ensemble(self, committed: str, options: list[WordOption],
                      logical_request_id: str):
        """Same options under `ensemble` cyclic rotations, in parallel; each
        answer is locally transformed (floor, temperature), summed by option
        display text and renormalized.  Returns (by_key, raws, stats)."""
        k = self.ensemble
        if k == 1:
            request = self._build_request(committed, options)
            outcome = self.provider.ask(request, logical_request_id, hook=self.hook)
            if outcome.status != "ok" or outcome.validated is None:
                raise RuntimeError(f"provider status {outcome.status}")
            answer = outcome.validated.answers.get("next_word")
            if answer is None or not answer.usable:
                raise RuntimeError("missing or unusable next_word answer")
            stats = (outcome.validated.returned_model,
                     outcome.attempts[-1].latency_ms if outcome.attempts else None,
                     outcome.usage_input_tokens)
            return answer.values.get("probabilities") or {}, None, stats

        n = len(options)
        jobs = []
        for i in range(k):
            offset = (i * n) // k
            ordered = options[offset:] + options[:offset]
            jobs.append((ordered, self._build_request(committed, ordered)))

        def ask_one(job):
            ordered, request = job
            outcome = self.provider.ask(request, f"{logical_request_id}.rot", hook=self.hook)
            return ordered, outcome

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(k, 8)) as pool_exec:
            results = list(pool_exec.map(ask_one, jobs))

        combined: dict[str, float] = {}
        raws = []
        model_returned = None
        latency = 0.0
        usage = 0
        for ordered, outcome in results:
            if outcome.status != "ok" or outcome.validated is None:
                raise RuntimeError(f"ensemble member status {outcome.status}")
            answer = outcome.validated.answers.get("next_word")
            if answer is None or not answer.usable:
                raise RuntimeError("missing or unusable next_word ensemble answer")
            raw = answer.values.get("probabilities") or {}
            raws.append(raw)
            model_returned = model_returned or outcome.validated.returned_model
            latency += outcome.attempts[-1].latency_ms if outcome.attempts else 0.0
            usage += outcome.usage_input_tokens or 0
            transformed = self.decoder.transform(raw)
            for j, o in enumerate(ordered):
                combined[o.text] = combined.get(o.text, 0.0) + transformed.get(f"opt_{j:03d}", 0.0)
            combined[END_TOKEN] = combined.get(END_TOKEN, 0.0) + transformed.get(END_TOKEN, 0.0)
        total = sum(combined.values())
        if total <= 0:
            raise RuntimeError("ensemble combined distribution is empty")
        combined = {key: value / total for key, value in combined.items()}
        by_key = {f"opt_{i:03d}": combined.get(o.text, 0.0) for i, o in enumerate(options)}
        by_key[END_TOKEN] = combined.get(END_TOKEN, 0.0)
        return by_key, raws, (model_returned, latency, usage)

    def _build_request(self, committed: str, options: list[WordOption]) -> SystemOneRequest:
        criteria = {}
        for i, o in enumerate(options):
            if o.is_end:
                continue
            if o.is_punct:
                criteria[f"opt_{i:03d}"] = f"punctuation: {o.text}"
            elif o.kind == "letter":
                criteria[f"opt_{i:03d}"] = f"append the letter, making {o.text!r}"
            else:
                criteria[f"opt_{i:03d}"] = f"write {o.text!r}"
        criteria[END_TOKEN] = "[end (finish the response)]"
        state = {
            "user_prompt": self.user_prompt,
            "previous_turns": self.previous_turns,
            "assistant_prefix": committed,
        }
        return SystemOneRequest(
            state=state, model=self.model,
            questions={"next_word": {
                "type": "choice",
                "instructions": self.instructions,
                "criteria": criteria,
            }})

    @staticmethod
    def _argmax(probs: dict[str, float], keys: list[str]) -> str:
        best_key, best_value = None, -1.0
        for key in keys:
            v = probs.get(key)
            if v is not None and v > best_value:
                best_value, best_key = v, key
        return best_key or END_TOKEN

    @staticmethod
    def _top(probs: dict[str, float], n: int) -> list[dict[str, Any]]:
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        return [{"key": k, "p": round(v, 6)} for k, v in ordered[:n]]

    @staticmethod
    def _repetition_stop(committed: str) -> bool:
        words = committed.lower().split()
        if len(words) >= 6 and len(set(words[-6:])) == 1:
            return True
        for unit_length in (12, 8, 4):
            unit = committed[-unit_length:]
            if unit and len(unit) * 4 <= len(committed) and committed.endswith(unit * 4):
                return True
        return False

    def _stop(self, reason: str, *, detail: str | None = None) -> dict[str, Any]:
        self.stop_reason = reason
        event: dict[str, Any] = {"type": "stop", "reason": reason}
        if detail:
            event["detail"] = detail
        self.events.append(event)
        return event

    def run(self) -> list[dict[str, Any]]:
        while self.stop_reason is None:
            self.step()
        return self.events

    # --------------------------------------------------------------- export
    def export_trace(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recorded_at": utc_now(),
            "user_prompt": self.user_prompt,
            "previous_turns": self.previous_turns,
            "decoder": {"mode": self.decoder.mode, "temperature": self.decoder.temperature,
                        "seed": self.decoder.seed, "top_p": self.decoder.top_p,
                        "top_k": self.decoder.top_k, "floor": self.decoder.floor},
            "limits": {"max_tokens": self.limits.max_tokens,
                       "max_seconds": self.limits.max_seconds},
            "stop_reason": self.stop_reason,
            "n_nodes": len(self.nodes),
            "prefix": self.prefix,
            "events": self.events,
            "nodes": [
                {k: v for k, v in node.items() if k != "parent_id"} | {
                    "parent_id": node["parent_id"]}
                for node in self.nodes.values()
            ],
        }
