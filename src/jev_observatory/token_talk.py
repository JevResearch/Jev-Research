"""Token-level Talk: Jev steers a local LM's next-token distribution.

New attempt after the character-level autoregression of ``talk.py`` (which
stays untouched).  Evidence base: ``docs/token-talk-plan.md`` and
``runs_live/talk_live_traces.json`` — greedy character picking degenerates
(``Geeee``, ``A    ``, ``AAB``) because the 98-character alphabet is a flat
candidate space.  Here a small *local* LM supplies linguistic competence
(softmax over its vocabulary, temperature 0, never sampled locally) and Jev
supplies the preference: at every step it chooses among the top-254 locally
likely next tokens plus END.

Honesty points (DESIGN.md §9 unchanged in spirit):

* this is NOT native chat and NOT "Jev talking": it is Jev re-scoring a local
  model's contextual next-token distribution, one Choice request per token;
* the local-LM distribution is a measurement artifact, recorded raw at every
  step alongside Jev's distribution and the decode rule actually used;
* option keys are opaque (``tok_000``…, ``END``) and never concatenated into
  output; option order is the local model's native yield order;
* the model's own EOS/special tokens are stripped from candidates — stopping
  is exclusively Jev's explicit END choice;
* stopping is conservative: END, token cap, wall clock, repetition guard,
  provider failure, cancellation, budget denial — never silent truncation.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .manifest import utc_now
from .providers import BaseProvider, NullHook
from .schema import SystemOneRequest

# ------------------------------------------------------------------ contract
END_TOKEN = "END"
MAX_OPTIONS = 254  # token candidates; +END = 255, the API contract cap


def creates_adjacent_repeat(
    current: str, addition: str, *, min_period: int = 3, max_period: int = 24,
) -> bool:
    """Would appending `addition` complete an adjacent doubled unit?

    Smarter anti-repetition: looks at the whole produced text, not just the
    last token.  A doubling of ANY unit with period in [min_period,
    max_period] is banned from the menu ("was was", "The the", "The
    wasthe. The wasthe."), while single/double letters stay writable
    ("letter", "hello"; min_period 3 exempts period-1 and period-2 units,
    so doubles like "ll" and pairs like "ha ha" mid-word are unaffected).
    The in-progress exemption (current already ends with all but the last
    character of the doubled unit) keeps legally-built pairs commitable:
    "ha ha" + " " is allowed even though "ha ha " is itself a doubling.
    Case-folded, so "The the" is caught.
    """
    new = (current + addition).lower()
    cur = current.lower()
    for candidate_text in (new, new + " "):  # virtual separator: catch units
        # that are only doubled once the next space arrives
        for period in range(min_period, max_period + 1):
            if len(candidate_text) < period * 2:
                break
            unit = candidate_text[-period:]
            doubled = unit * 2
            if candidate_text.endswith(doubled) and not cur.endswith(doubled[:-1]):
                return True
    return False

NEXT_TOKEN_INSTRUCTIONS = (
    "Select the next token of the assistant response, conditioned on this "
    "conversation transcript and the exact assistant prefix shown. The options "
    "are the locally most likely next tokens, most likely first. Choose END "
    "when the response is complete."
)
END_DESCRIPTION = "end the assistant response now"


@dataclass(frozen=True)
class TokenCandidate:
    token_id: int
    text: str
    p_local: float
    injected: str | None = None   # identity-parity provenance (findings §5/16)


# Common model and manufacturer names for identity-parity intervention: when
# a high-ranked candidate names a model (or maker), the leading token(s) for
# "Jev" (or "Typesafe") are injected at the same probability, so Jev can
# always express its own name if asked about models.
IDENTITY_MODEL_NAMES = frozenset({
    "gpt", "chatgpt", "claude", "gemini", "llama", "mistral", "copilot",
    "deepseek", "grok", "qwen", "bert", "glm",
})
IDENTITY_MAKER_NAMES = frozenset({
    "openai", "anthropic", "google", "microsoft", "meta", "nvidia",
    "deepmind", "huggingface", "jetbrains", "xai",
})
IDENTITY_TARGET_MODEL = "Jev"
IDENTITY_TARGET_MAKER = "Typesafe"
SECONDARY_FACTOR = 0.5  # other missing names enter at half the reference p


def _name_fragment(text: str) -> str:
    return text.strip().strip("-").lower()


def _matches_name(text: str, names: frozenset[str]) -> bool:
    frag = _name_fragment(text)
    if not frag:
        return False
    if frag in names:
        return True
    return len(frag) >= 4 and any(n.startswith(frag) for n in names)


def identity_intervention(
    candidates: list[TokenCandidate],
    *,
    prefix: str,
    spell: Callable[[str], list[tuple[int, str]]],
    max_total: int = MAX_OPTIONS,
) -> tuple[list[TokenCandidate], list[TokenCandidate]]:
    """Identity-parity injection.

    When a relatively high-ranked candidate names a common model, the
    leading token(s) spelling "Jev" are injected at the SAME probability
    (continuation tokens are injected at later steps so the word can
    complete); other common model names missing from the menu enter at a
    lower probability.  Same mechanism for manufacturers vs "Typesafe".
    Returns (merged candidates, injected list).
    """
    injected: list[TokenCandidate] = []
    present_ids = {c.token_id for c in candidates}
    present_texts = {_name_fragment(c.text) for c in candidates}

    def ensure_spelling(target: str, ref_p: float, tag: str) -> None:
        pieces = spell(target)
        if not pieces:
            return
        words = prefix.split()
        tail = words[-1] if words else ""
        started = bool(tail) and target.startswith(tail) and tail != target
        consumed = len(tail) if started else 0
        for tid, text in pieces:
            if consumed >= len(text):
                consumed -= len(text)
                continue
            if tid in present_ids or _name_fragment(text) in present_texts:
                return  # already offered; later pieces arrive next step
            injected.append(TokenCandidate(
                token_id=tid, text=text, p_local=ref_p, injected=tag))
            present_ids.add(tid)
            return

    def secondary(name: str, ref_p: float, tag: str) -> None:
        pieces = spell(name)
        if not pieces:
            return
        tid, text = pieces[0]
        if tid not in present_ids and _name_fragment(text) not in present_texts:
            injected.append(TokenCandidate(
                token_id=tid, text=text, p_local=ref_p * SECONDARY_FACTOR,
                injected=f"identity:{name}"))
            present_ids.add(tid)

    model_hits = [c for c in candidates if _matches_name(c.text, IDENTITY_MODEL_NAMES)]
    if model_hits:
        ref_p = max(c.p_local for c in model_hits)
        ensure_spelling(IDENTITY_TARGET_MODEL, ref_p, "identity:Jev")
        for name in sorted(IDENTITY_MODEL_NAMES):
            if name not in present_texts and name != IDENTITY_TARGET_MODEL.lower():
                secondary(name, ref_p, f"identity:{name}")
    maker_hits = [c for c in candidates if _matches_name(c.text, IDENTITY_MAKER_NAMES)]
    if maker_hits:
        ref_p = max(c.p_local for c in maker_hits)
        ensure_spelling(IDENTITY_TARGET_MAKER, ref_p, "identity:Typesafe")
        for name in sorted(IDENTITY_MAKER_NAMES):
            if name not in present_texts and name != IDENTITY_TARGET_MAKER.lower():
                secondary(name, ref_p, f"identity:{name}")

    if not injected:
        return candidates, []
    merged = list(candidates) + injected
    merged.sort(key=lambda c: (-c.p_local, c.token_id))
    budget = max_total - 1  # END
    overflow = len(merged) - budget
    if overflow > 0:
        kept = []
        for c in merged:
            if overflow and c.injected is None:
                overflow -= 1
                continue
            kept.append(c)
        merged = kept
    return merged, injected



class LocalLM(Protocol):
    """A local language model used purely as a next-token scorer.

    ``prepare`` resets state and ingests the conversation; ``advance`` appends
    one chosen token; ``score`` returns the full sorted next-token
    distribution (native order: descending probability).  The LM never
    samples; it supplies candidates and likelihoods only.
    """

    name: str
    special_token_ids: frozenset[int]

    def prepare(self, *, user_prompt: str, previous_turns: list[dict[str, str]]) -> None: ...

    def advance(self, token_id: int) -> None: ...

    def score(self) -> list[TokenCandidate]: ...

    def save_state(self) -> bytes: ...

    def load_state(self, state: bytes) -> None: ...


class ScriptedLM:
    """Deterministic test fixture: a fixed list of candidate tables, one per
    scoring call. ``advance`` is recorded so tests can assert exactly which
    tokens were fed back. ``save_state``/``load_state`` capture the cursor."""

    def __init__(self, steps: list[list[TokenCandidate]], spellings: dict[str, list[tuple[int, str]]] | None = None) -> None:
        self.name = "scripted-lm"
        self.special_token_ids = frozenset({999_999})
        self.steps = steps
        self.position = 0
        self.fed: list[int] = []
        self.prepared: list[dict[str, Any]] = []
        self.spellings = spellings or {}

    def prepare(self, *, user_prompt: str, previous_turns: list[dict[str, str]]) -> None:
        self.position = 0
        self.fed = []
        self.prepared.append({"user_prompt": user_prompt, "previous_turns": previous_turns})

    def advance(self, token_id: int) -> None:
        self.fed.append(token_id)
        self.position += 1

    def score(self, limit: int | None = None) -> list[TokenCandidate]:
        if self.position >= len(self.steps):
            return []
        return list(self.steps[self.position])

    def spell_word(self, word: str) -> list[tuple[int, str]]:
        return self.spellings.get(word, [])

    def save_state(self) -> bytes:
        return str(self.position).encode()

    def load_state(self, state: bytes) -> None:
        self.position = int(state.decode())


# ------------------------------------------------------------- candidate set
def _text_is_representable(text: str) -> bool:
    """Reject tokens decoding to control characters or chat-template markers
    (special-token text can render literally under ``special=False``)."""
    if "<|" in text or "|>" in text:
        return False
    for char in text:
        if char in "\n\t":
            continue
        if unicodedata.category(char) == "Cc":
            return False
    return True


def build_candidates(
    scored: list[TokenCandidate], *, special_token_ids: frozenset[int] = frozenset(),
    banned_texts: frozenset[str] = frozenset(),
    max_options: int = MAX_OPTIONS,
) -> list[TokenCandidate]:
    """Filter and cap the model's native-order score list.

    Drops EOS/special token ids, control-character text, tokens whose word
    form is in ``banned_texts`` (recent repetitions are never offered), and
    keeps the model's native yield order (descending local probability),
    capped at 254 so the request fits the 255-option contract limit together
    with END.
    """
    out: list[TokenCandidate] = []
    for candidate in scored:
        if candidate.token_id in special_token_ids:
            continue
        if not candidate.text or not _text_is_representable(candidate.text):
            continue
        if banned_texts and (candidate.text in banned_texts or (
                candidate.text.strip() and candidate.text.strip() in banned_texts)):
            continue
        out.append(candidate)
        if len(out) >= max_options:
            break
    return out


def candidate_criteria(candidates: list[TokenCandidate]) -> dict[str, str]:
    """Choice criteria: opaque visible keys -> descriptions. Keys are never
    concatenated into output; token text is inert, repr-rendered data."""
    criteria: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        criteria[f"tok_{index:03d}"] = f"the token {candidate.text!r}"
    criteria[END_TOKEN] = END_DESCRIPTION
    return criteria


def build_token_talk_request(
    *, user_prompt: str, previous_turns: list[dict[str, str]], prefix: str,
    candidates: list[TokenCandidate], model: str = "jev-1.13.0",
    include_end: bool = True,
) -> SystemOneRequest:
    """One token-choice request. Transcript and prefix are distinct fields."""
    state = {
        "user_prompt": user_prompt,
        "previous_turns": previous_turns,
        "assistant_prefix": prefix,
    }
    criteria = candidate_criteria(candidates)
    if not include_end:
        criteria = {k: v for k, v in criteria.items() if k != END_TOKEN}
    return SystemOneRequest(
        state=state,
        model=model,
        questions={"next_token": {
            "type": "choice",
            "instructions": NEXT_TOKEN_INSTRUCTIONS,
            "criteria": criteria,
        }},
    )


# ------------------------------------------------------------------ decoding
DECODER_MODES = ("local_greedy", "jev_greedy", "jev_sample", "product")


@dataclass(frozen=True)
class DecoderConfig:
    mode: str = "jev_greedy"        # local_greedy | jev_greedy | jev_sample | product
    temperature: float | None = None  # local transform; recorded, never hidden
    seed: int | None = None
    top_p: float | None = None       # nucleus cutoff on the transformed distribution
    top_k: int | None = None         # keep the k largest after temperature
    floor: float | None = None       # subtract the quantization floor, renormalize
    ensemble: int = 1                # >1: same options under K cyclic rotations, add+norm

    def __post_init__(self) -> None:
        if self.mode not in DECODER_MODES:
            raise ValueError(f"unknown decoder mode {self.mode!r}")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.floor is not None and not 0.0 < self.floor < 1.0:
            raise ValueError("floor must be in (0, 1)")
        if self.ensemble < 1:
            raise ValueError("ensemble must be >= 1")

    def transform(self, probs: dict[str, float]) -> dict[str, float]:
        """Local transforms, in order: floor subtraction, temperature.

        The vendor quantizes probabilities (observed resolution); options at
        the quantization floor carry no ranking signal, so the floor is
        subtracted and the remainder renormalized before anything else.
        """
        if self.floor is not None:
            trimmed = {key: max(0.0, value - self.floor) for key, value in probs.items()}
            total = sum(trimmed.values())
            if total > 0:
                probs = {key: value / total for key, value in trimmed.items()}
        if not self.temperature or self.temperature == 1.0:
            return dict(probs)
        temperature = self.temperature
        transformed = {
            key: (value ** (1.0 / temperature) if value > 0 else 0.0)
            for key, value in probs.items()
        }
        total = sum(transformed.values())
        if total <= 0:
            return dict(probs)
        return {key: value / total for key, value in transformed.items()}

    def truncate(self, probs: dict[str, float]) -> dict[str, float]:
        """Nucleus/top-k truncation on the transformed distribution.

        A local operation on Jev's answer: Jev still sees and scores all
        options; this only bounds the *sampler*.  The raw values stay
        recorded; this result is the 'transformed' distribution.
        """
        if self.top_p is None and self.top_k is None:
            return probs
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        kept: list[tuple[str, float]] = []
        cumulative = 0.0
        for index, (key, value) in enumerate(ordered):
            if self.top_k is not None and index >= self.top_k:
                break
            kept.append((key, value))
            cumulative += value
            if self.top_p is not None and cumulative >= self.top_p:
                break
        total = sum(value for _, value in kept)
        if total <= 0:
            return probs
        return {key: value / total for key, value in kept}


def _argmax(probs: dict[str, float], keys: list[str]) -> str:
    """Argmax with the fixed tie rule: first key in candidate order."""
    best_key = None
    best_value = -1.0
    for key in keys:
        value = probs.get(key)
        if value is not None and value > best_value:
            best_value = value
            best_key = key
    return best_key or END_TOKEN


# ------------------------------------------------------------------ session
@dataclass
class TokenTalkLimits:
    max_tokens: int = 64
    max_seconds: float = 240.0
    max_provider_requests: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass
class _Node:
    node_id: str
    parent_id: str | None
    token_id: int | None
    text: str | None
    key: str | None
    candidates: list[TokenCandidate] | None
    jev_probabilities: dict[str, float] | None
    transformed: dict[str, float] | None
    ensemble_raw: list[dict[str, float]] | None  # per-ordering raw (ensemble mode)
    local_choice_key: str  # what the local LM alone would have picked
    lm_state: bytes | None  # local LM state after this node's choice
    user_edited: bool
    model_returned: str | None = None
    latency_ms: float | None = None
    usage_input_tokens: int | None = None


@dataclass
class TokenTalkSession:
    """One generation session: a tree of token nodes with explicit stops."""

    session_id: str
    provider: BaseProvider
    local_lm: LocalLM
    user_prompt: str
    previous_turns: list[dict[str, str]] = field(default_factory=list)
    limits: TokenTalkLimits = field(default_factory=TokenTalkLimits)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    candidate_order: str = "native"      # native | reversed | shuffled
    extra_token_ids: tuple[int, ...] = ()  # appended before END (injection probe)
    max_options: int = MAX_OPTIONS       # expanded choices: total pool across pages
    anti_repeat: bool = True             # structural adjacent-doubling ban
    identity_parity: bool = True         # inject Jev/Typesafe spellings at model-name parity
    hook: Any = None
    model: str = "jev-1.13.0"
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.nodes: dict[str, _Node] = {}
        self.current_node_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self.cancelled = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._rng = None
        if self.decoder.mode in ("jev_sample", "product") and self.decoder.seed is not None:
            self._rng = __import__("random").Random(self.decoder.seed)
        self.hook = self.hook or NullHook()
        self._provider_requests = 0
        self.local_lm.prepare(user_prompt=self.user_prompt, previous_turns=self.previous_turns)

    # ------------------------------------------------------------- state
    @property
    def prefix(self) -> str:
        texts: list[str] = []
        node_id = self.current_node_id
        while node_id is not None:
            node = self.nodes[node_id]
            if node.text is not None:
                texts.append(node.text)
            node_id = node.parent_id
        return "".join(reversed(texts))

    @property
    def token_ids(self) -> list[int]:
        ids: list[int] = []
        node_id = self.current_node_id
        while node_id is not None:
            node = self.nodes[node_id]
            if node.token_id is not None:
                ids.append(node.token_id)
            node_id = node.parent_id
        return list(reversed(ids))

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return self.clock() - self.started_at

    def cancel(self) -> None:
        self.cancelled = True

    # -------------------------------------------------------------- stepping
    def step(self) -> dict[str, Any]:
        """Take one decoding step; returns the event (with stop info if stopping)."""
        if self.stop_reason is not None:
            return {"type": "stop", "reason": self.stop_reason, "already_stopped": True}
        if self.cancelled:
            return self._stop("cancelled")
        if self.started_at is None:
            self.started_at = self.clock()
        if self._elapsed() > self.limits.max_seconds:
            return self._stop("deadline")
        if len(self.nodes) >= self.limits.max_tokens:
            return self._stop("max_tokens")
        if (
            self.limits.max_provider_requests is not None
            and self._provider_requests >= self.limits.max_provider_requests
        ):
            return self._stop("max_provider_requests")

        scored = self.local_lm.score(self.max_options + 64)
        candidates = build_candidates(
            scored, special_token_ids=self.local_lm.special_token_ids,
            banned_texts=self._recent_words(),
            max_options=self.max_options - len(self.extra_token_ids),
        )
        if self.anti_repeat:
            before = len(candidates)
            candidates = [
                c for c in candidates
                if not creates_adjacent_repeat(self.prefix, c.text)
            ]
            if before != len(candidates):
                self.events.append({"type": "anti_repeat", "dropped": before - len(candidates)})
        if self.identity_parity:
            spell = getattr(self.local_lm, "spell_word", None)
            if spell is not None:
                candidates, injected = identity_intervention(
                    candidates, prefix=self.prefix, spell=spell,
                    max_total=self.max_options)
                if injected:
                    self.events.append({"type": "identity_injection", "count": len(injected),
                                        "options": [c.text for c in injected],
                                        "p": [round(c.p_local, 6) for c in injected]})
        candidates = self._apply_order(candidates)
        if self.extra_token_ids:
            candidates = list(candidates) + [
                TokenCandidate(token_id=tid, text=self._extra_text(tid), p_local=0.0)
                for tid in self.extra_token_ids
            ]
        # the model yielding nothing usable leaves only Jev's END option
        if not candidates:
            return self._stop("no_candidates")
        if self._token_repetition_stop():
            return self._stop("repetition")
        if self._text_repetition_stop(self.prefix):
            return self._stop("repetition")

        self._provider_requests += max(1, self.decoder.ensemble)
        try:
            raw_probs, ensemble_raw, stats = self._ask_ensemble(
                candidates,
                f"talk-tokens:{self.session_id}:{len(self.events)}",
            )
        except Exception as exc:
            from .guards import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                return self._stop("budget_exceeded")
            return self._stop("provider_failure", detail=f"exception:{type(exc).__name__}")
        key_to_candidate = {f"tok_{i:03d}": c for i, c in enumerate(candidates)}
        keys = list(key_to_candidate) + [END_TOKEN]

        local_choice_key = "tok_000" if candidates else END_TOKEN
        chosen_key, transformed = self._decode(raw_probs, key_to_candidate, keys)

        candidate = key_to_candidate.get(chosen_key)
        token_id = candidate.token_id if candidate else None
        text = candidate.text if candidate else None

        node = _Node(
            node_id=f"n{len(self.nodes):04d}",
            parent_id=self.current_node_id,
            token_id=token_id,
            text=text,
            key=chosen_key,
            candidates=candidates,
            jev_probabilities=raw_probs,
            transformed=transformed if transformed != raw_probs else None,
            ensemble_raw=ensemble_raw,
            local_choice_key=local_choice_key,
            lm_state=None,
            user_edited=False,
            model_returned=stats[0],
            latency_ms=stats[1],
            usage_input_tokens=stats[2],
        )
        self.nodes[node.node_id] = node
        self.current_node_id = node.node_id

        event = {
            "type": "step",
            "node_id": node.node_id,
            "key": chosen_key,
            "token_id": token_id,
            "text": text,
            "local_top": [
                {"token_id": c.token_id, "text": c.text, "p": round(c.p_local, 6)}
                for c in candidates[:5]
            ],
            "jev_top": self._top(raw_probs, key_to_candidate, 5),
            "local_choice": local_choice_key,
            "model_returned": node.model_returned,
            "latency_ms": node.latency_ms,
        }
        self.events.append(event)
        if token_id is None:
            node.lm_state = self.local_lm.save_state()  # unchanged by END
            return self._stop("end_token")
        self.local_lm.advance(token_id)
        node.lm_state = self.local_lm.save_state()
        return event

    def _ask_ensemble(self, candidates, logical_request_id):
        """Expanded choices x rotation ensemble, dispatched in parallel.

        The candidate pool is split into pages of <=254 options (contract
        cap; END rides only on the first page).  Each page is presented
        under `ensemble` cyclic rotations; every answer is locally
        transformed (floor, temperature), then all pages and rotations are
        summed by token text and renormalized into one big vector.
        Returns (combined-by-key, per-request raw list, (model, latency, usage)).
        """
        k = max(1, self.decoder.ensemble)
        pool = list(candidates)
        pages = [pool[i:i + MAX_OPTIONS] for i in range(0, len(pool), MAX_OPTIONS)] or [[]]

        jobs = []
        for p_index, page in enumerate(pages):
            n = len(page)
            for i in range(k):
                offset = (i * n) // k if n else 0
                ordered = page[offset:] + page[:offset]
                request = build_token_talk_request(
                    user_prompt=self.user_prompt, previous_turns=self.previous_turns,
                    prefix=self.prefix, candidates=ordered, model=self.model,
                    include_end=(p_index == 0),
                )
                jobs.append((p_index, ordered, request))
        if len(jobs) == 1:
            outcome = self.provider.ask(jobs[0][2], logical_request_id, hook=self.hook)
            if outcome.status != "ok" or outcome.validated is None:
                raise RuntimeError(f"provider status {outcome.status}")
            answer = outcome.validated.answers.get("next_token")
            if answer is None or not answer.usable:
                raise RuntimeError("missing or unusable next_token answer")
            stats = (outcome.validated.returned_model,
                     outcome.attempts[-1].latency_ms if outcome.attempts else None,
                     outcome.usage_input_tokens)
            return answer.values.get("probabilities") or {}, None, stats

        def ask_one(job):
            p_index, ordered, request = job
            outcome = self.provider.ask(
                request, f"{logical_request_id}.p{p_index}", hook=self.hook)
            return p_index, ordered, outcome

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(jobs), 8)) as pool_exec:
            results = list(pool_exec.map(ask_one, jobs))

        combined: dict[str, float] = {}
        raws: list[dict[str, float]] = []
        model_returned = None
        latency = 0.0
        usage = 0
        for p_index, ordered, outcome in results:
            if outcome.status != "ok" or outcome.validated is None:
                raise RuntimeError(f"ensemble member status {outcome.status}")
            answer = outcome.validated.answers.get("next_token")
            if answer is None or not answer.usable:
                raise RuntimeError("missing or unusable next_token ensemble answer")
            raw = answer.values.get("probabilities") or {}
            raws.append(raw)
            model_returned = model_returned or outcome.validated.returned_model
            latency += outcome.attempts[-1].latency_ms if outcome.attempts else 0.0
            usage += outcome.usage_input_tokens or 0
            transformed = self.decoder.transform(raw)
            for j, c in enumerate(ordered):
                combined[c.text] = combined.get(c.text, 0.0) + transformed.get(f"tok_{j:03d}", 0.0)
            if p_index == 0:
                combined[END_TOKEN] = combined.get(END_TOKEN, 0.0) + transformed.get(END_TOKEN, 0.0)
        total = sum(combined.values())
        if total <= 0:
            raise RuntimeError("ensemble combined distribution is empty")
        combined = {key: value / total for key, value in combined.items()}
        # re-express combined over the base pool's keys for _decode
        by_key = {f"tok_{j:03d}": combined.get(c.text, 0.0)
                  for j, c in enumerate(candidates)}
        by_key[END_TOKEN] = combined.get(END_TOKEN, 0.0)
        return by_key, raws, (model_returned, latency, usage)

    def _recent_words(self, depth: int = 4) -> frozenset[str]:
        """Word forms used in the last `depth` tokens: never offered again.

        Repeated words cannot cause drift if Jev cannot choose them; the
        filtering is visible in each node's candidate list and the word-×6
        guard remains as a backstop.
        """
        words: set[str] = set()
        node_id = self.current_node_id
        seen = 0
        while node_id is not None and seen < depth:
            text = self.nodes[node_id].text
            if text:
                words.update(w for w in text.split() if w)
                seen += 1
            node_id = self.nodes[node_id].parent_id
        return frozenset(words)

    def _apply_order(self, candidates: list[TokenCandidate]) -> list[TokenCandidate]:
        """Order-mode transform, applied AFTER filtering; ordering is recorded."""
        if self.candidate_order == "native":
            return candidates
        if self.candidate_order == "reversed":
            return list(reversed(candidates))
        if self.candidate_order == "shuffled":
            rng = __import__("random").Random(
                f"{self.decoder.seed or 0}:{len(self.nodes)}"
            )
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            return shuffled
        raise ValueError(f"unknown candidate_order {self.candidate_order!r}")

    def _extra_text(self, token_id: int) -> str:
        text = self.local_lm.detokenize_text(token_id)
        return text if text is not None else f"<id {token_id}>"

    def _token_repetition_stop(self) -> bool:
        """Same token id chosen 4 times in a row: degenerate loop."""
        ids = self.token_ids
        return len(ids) >= 4 and len(set(ids[-4:])) == 1

    def _decode(
        self,
        raw_probs: dict[str, float],
        key_to_candidate: dict[str, TokenCandidate],
        keys: list[str],
    ) -> tuple[str, dict[str, float]]:
        transformed = self.decoder.transform(raw_probs)
        mode = self.decoder.mode
        if mode == "local_greedy":
            # Jev's answer is recorded but never used for the choice: the
            # candidate order IS the local ranking (native yield order), so
            # the local argmax is the first candidate.
            return ("tok_000" if key_to_candidate else END_TOKEN), transformed
        if mode == "jev_greedy":
            return _argmax(transformed, keys), transformed
        weights = transformed
        if mode == "product":
            # END's local prior is the model's own EOS probability, not 1.0:
            # an unweighted END wins every flat, high-entropy position.
            end_prior = getattr(self.local_lm, "stop_p_local", 1.0)
            weights = {
                key: transformed.get(key, 0.0) * (
                    key_to_candidate[key].p_local if key in key_to_candidate else end_prior
                )
                for key in keys
            }
            total = sum(weights.values())
            if total <= 0:
                weights = transformed
            else:
                weights = {key: value / total for key, value in weights.items()}
        # sampler truncation (top_p / top_k) applies to jev_sample and product
        weights = self.decoder.truncate(weights)
        if self._rng is not None and list(weights.values()):
            population = list(weights)
            chosen = self._rng.choices(
                population=population, weights=list(weights.values()), k=1
            )[0]
            return chosen, weights
        return _argmax(weights, keys), weights

    def _stop(self, reason: str, *, detail: str | None = None) -> dict[str, Any]:
        self.stop_reason = reason
        self.finished_at = self.clock()
        event: dict[str, Any] = {"type": "stop", "reason": reason, "prefix_tokens": len(self.nodes)}
        if detail:
            event["detail"] = detail
        self.events.append(event)
        return event

    @staticmethod
    def _top(probs: dict[str, float], key_to_candidate: dict[str, TokenCandidate],
             n: int) -> list[dict[str, Any]]:
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        return [{"key": key, "text": key_to_candidate[key].text if key in key_to_candidate else None,
                 "p": round(value, 6)} for key, value in ordered[:n]]

    @staticmethod
    def _text_repetition_stop(prefix: str) -> bool:
        """Conservative loop guard: a short unit repeated at least 4 times at
        the tail, or one whitespace-word repeated 6+ times in the tail."""
        for unit_length in (12, 8, 4, 2):
            unit = prefix[-unit_length:]
            if unit and len(unit) * 4 <= len(prefix) and prefix.endswith(unit * 4):
                return True
        words = prefix.split()
        if len(words) >= 6:
            tail = words[-6:]
            if len(set(tail)) == 1:
                return True
        return False

    # ------------------------------------------------------------ traversal
    def run(self) -> list[dict[str, Any]]:
        """Decode until a stop condition; cancellation is checked between steps."""
        while self.stop_reason is None:
            self.step()
        return self.events

    def backtrack(self, node_id: str) -> None:
        """Move the cursor to an ancestor node; branches are never deleted."""
        if node_id not in self.nodes:
            raise KeyError(f"unknown node {node_id}")
        walker = self.current_node_id
        while walker is not None:
            if walker == node_id:
                self.current_node_id = node_id
                node = self.nodes[node_id]
                if node.lm_state is not None:
                    self.local_lm.load_state(node.lm_state)
                self.stop_reason = None  # a backtrack re-opens a stopped session
                return
            walker = self.nodes[walker].parent_id
        raise ValueError(f"node {node_id} is not an ancestor of the current position")

    def export_trace(self) -> dict[str, Any]:
        path: list[str] = []
        walker = self.current_node_id
        while walker is not None:
            path.append(walker)
            walker = self.nodes[walker].parent_id
        return {
            "session_id": self.session_id,
            "recorded_at": utc_now(),
            "user_prompt": self.user_prompt,
            "previous_turns": self.previous_turns,
            "local_lm": self.local_lm.name,
            "decoder": {"mode": self.decoder.mode, "temperature": self.decoder.temperature,
                        "seed": self.decoder.seed, "top_p": self.decoder.top_p,
                        "top_k": self.decoder.top_k, "floor": self.decoder.floor},
            "limits": {"max_tokens": self.limits.max_tokens,
                       "max_seconds": self.limits.max_seconds},
            "stop_reason": self.stop_reason,
            "n_nodes": len(self.nodes),
            "active_path_nodes": list(reversed(path)),
            "prefix": self.prefix,
            "token_ids": self.token_ids,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "token_id": node.token_id,
                    "text": node.text,
                    "key": node.key,
                    "user_edited": node.user_edited,
                    "candidates": [
                        {"token_id": c.token_id, "text": c.text,
                         "p_local": round(c.p_local, 8),
                         **({"injected": c.injected} if c.injected else {})}
                        for c in (node.candidates or [])
                    ],
                    "jev_probabilities": node.jev_probabilities,
                    "transformed_probabilities": node.transformed,
                    "ensemble_raw": node.ensemble_raw,
                    "local_choice_key": node.local_choice_key,
                    "model_returned": node.model_returned,
                    "latency_ms": node.latency_ms,
                    "usage_input_tokens": node.usage_input_tokens,
                }
                for node in self.nodes.values()
            ],
            "events": self.events,
        }


class ScriptedTokenProvider:
    """Deterministic test fixture: answers `next_token` from a script of
    (choice, probabilities) pairs; raises ValueError when exhausted."""

    name = "scripted-token"

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.answers = answers
        self._position = 0
        self.requests: list[SystemOneRequest] = []

    def ask(self, request: SystemOneRequest, logical_request_id: str, **kwargs) -> Any:
        self.requests.append(request)
        from .providers import AttemptRecord, CallOutcome
        from .validation import validate_response

        if self._position >= len(self.answers):
            raise ValueError("scripted provider exhausted")
        answer = dict(self.answers[self._position])
        self._position += 1
        # fixture convenience: fill the distribution over ALL offered options
        # (the live server does this too); unmatched keys get 0.0.
        answer = {**answer}
        q = request.questions["next_token"]
        probs = dict(answer.get("probabilities") or {})
        for key in q.criteria:
            probs.setdefault(key, 0.0)
        answer["probabilities"] = probs
        answer.setdefault("confidence", 0.5)
        body = {
            "model": request.model,
            "answers": {"next_token": {"type": "choice", **answer}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        validated = validate_response(body, request, http_status=200)
        record = AttemptRecord(
            attempt_id=f"{logical_request_id}.0", logical_request_id=logical_request_id,
            attempt_index=0, provider=self.name, model_requested=request.model,
            started_at=utc_now(), finished_at=utc_now(), latency_ms=1.0, first_byte_ms=None,
            cold_connection=False, http_status=200, outcome="ok", uncertain=False, error=None,
            usage_input_tokens=10, usage_output_tokens=2, estimated_input_tokens=10,
            response_bytes=len(json.dumps(body)), request_sha256=request.request_sha256(),
        )
        return CallOutcome(
            logical_request_id=logical_request_id, status="ok", terminal=True,
            attempts=[record], validated=validated, usage_input_tokens=10,
            usage_output_tokens=2, total_latency_ms=1.0, n_retries=0,
            request_sha256=request.request_sha256(), note="scripted-token",
        )

    def close(self) -> None:
        return
