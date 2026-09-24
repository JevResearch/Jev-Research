"""Local-LM backend for token-level Talk: llama-cpp-python over a GGUF.

The local model is used *only* as a next-token scorer: prepare ingests the
formatted conversation, advance appends one chosen token id, score returns
the softmax over the full vocabulary in native (descending) order.  No local
sampling ever happens; Jev is the sole decision-maker.

Kept in a separate module so the core session (``token_talk.py``) has no
llama-cpp dependency; tests run entirely on ``ScriptedLM``.
"""

from __future__ import annotations

from typing import Any

from .token_talk import TokenCandidate


class ContextLimitExceeded(RuntimeError):
    """The formatted conversation no longer fits the local model's context."""


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [pow(2.718281828459045, value - top) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


class LlamaCppLM:
    """LocalLM implementation on llama-cpp-python with GPU offload."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        from llama_cpp import Llama  # optional dependency; lazy import

        self.name = f"llama-cpp:{model_path.rsplit('/', 1)[-1]}"
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self._n_ctx = n_ctx
        self._formatter = self._build_formatter()
        self._blocked_texts = self._template_marker_fragments()
        self._last_logits: list[float] | None = None
        self._stop_p_local = 1.0
        # special tokens are never offered to Jev: stopping is Jev's END choice
        self.special_token_ids = frozenset({self._llm.token_eos(), self._llm.token_bos()})

    # ----------------------------------------------------------- template
    def _build_formatter(self) -> Any:
        """A chat formatter built from the GGUF's own chat template."""
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        template = self._llm.metadata.get("tokenizer.chat_template")
        if template is None:
            raise ValueError("model GGUF has no tokenizer.chat_template; cannot format chat")
        return Jinja2ChatFormatter(
            template=template,
            eos_token=self._llm.detokenize([self._llm.token_eos()], special=True).decode(
                "utf-8", errors="replace"
            ),
            bos_token=self._llm.detokenize([self._llm.token_bos()], special=True).decode(
                "utf-8", errors="replace"
            ),
            add_generation_prompt=True,
        )

    def _render_prompt(self, *, user_prompt: str, previous_turns: list[dict[str, str]],
                       prefix: str) -> str:
        """Template-rendered prompt whose tail is the partial assistant reply.

        The closing suffix the template appends after an assistant message is
        measured with a marker message and stripped, so the prompt ends exactly
        at the end of the partial prefix (an open assistant turn).
        """
        messages: list[dict[str, str]] = [
            {"role": turn.get("role", "user"), "content": turn.get("text", "")}
            for turn in previous_turns
        ]
        messages.append({"role": "user", "content": user_prompt})
        base = self._formatter(messages=messages).prompt
        if not prefix:
            return base
        marker = "\u0001jev-marker\u0001"
        with_prefix = self._formatter(
            messages=messages + [{"role": "assistant", "content": prefix + marker}]
        ).prompt
        idx = with_prefix.find(marker)
        if idx < 0:
            raise ValueError("chat template dropped or transformed the marker")
        # everything after the marker is the template's post-assistant closing;
        # the open assistant turn's prompt is base + prefix + (any closing text
        # the template places between content and marker position)
        prefix_part = with_prefix[:idx]
        closing = _common_suffix(prefix_part, base)
        return prefix_part[: len(prefix_part) - len(closing)]

    def _template_marker_fragments(self) -> frozenset[str]:
        """Fragments of the template's own control markers (e.g. <|im_end|>).

        With the model's EOS stripped from candidates, positions after the
        natural end offer marker *pieces* ('|<', 'im_end', '|>') as ordinary
        tokens.  They must never be speakable, so every fragment of every
        marker found in the template is blocked (single ASCII angle/pipe
        characters included)."""
        import re

        template = self._llm.metadata.get("tokenizer.chat_template", "")
        markers = re.findall(r"<\|[^|<>]*\|>", template)
        markers += [
            self._llm.detokenize([t], special=True).decode("utf-8", errors="replace")
            for t in (self._llm.token_eos(), self._llm.token_bos())
        ]
        blocked: set[str] = set()
        for marker in set(markers):
            if not marker:
                continue
            for width in range(1, len(marker) + 1):
                for start in range(len(marker) - width + 1):
                    blocked.add(marker[start : start + width])
        return frozenset(blocked)

    # -------------------------------------------------------------- LocalLM
    def prepare(self, *, user_prompt: str, previous_turns: list[dict[str, str]]) -> None:
        self._llm.reset()  # a fresh session must not inherit prior KV state
        prompt = self._render_prompt(
            user_prompt=user_prompt, previous_turns=previous_turns, prefix="",
        )
        tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)
        if len(tokens) + 2 >= self._n_ctx:
            raise ContextLimitExceeded(
                f"prompt of {len(tokens)} tokens exceeds n_ctx={self._n_ctx}"
            )
        self._llm.eval(tokens)
        self._cache_logits()

    def _cache_logits(self) -> None:
        # the context keeps logits for the last decoded position only; copy
        # before the next decode reuses the buffer.
        import numpy as np

        n_vocab = self._llm.n_vocab()
        self._last_logits = np.ctypeslib.as_array(
            self._llm._ctx.get_logits(), shape=(n_vocab,)
        ).tolist()

    def advance(self, token_id: int) -> None:
        self._llm.eval([token_id])
        self._cache_logits()

    def score(self, limit: int | None = None) -> list[TokenCandidate]:
        from .token_talk import MAX_OPTIONS

        limit = limit if limit is not None else MAX_OPTIONS + 64
        if self._last_logits is None:
            raise RuntimeError("no logits available: prepare/advance not called")
        probs = _softmax(self._last_logits)
        # the model's own EOS probability: END's local prior in product mode
        # (END is not a candidate, so without this it would carry 1.0)
        self._stop_p_local = probs[self._llm.token_eos()]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        out: list[TokenCandidate] = []
        for token_id in order:
            text = self._llm.detokenize([token_id], special=False).decode(
                "utf-8", errors="replace"
            )
            if text in self._blocked_texts:
                continue
            out.append(TokenCandidate(token_id=token_id, text=text, p_local=probs[token_id]))
            if len(out) >= limit + 64:  # pre-filter headroom for empties/controls
                break
        return out

    @property
    def stop_p_local(self) -> float:
        """The local model's own EOS probability at the current position."""
        return self._stop_p_local

    def _softmax_current(self) -> list[float]:
        """Softmax over the current logits (sweep-time scoring helper)."""
        return _softmax(self._last_logits)

    def spell_word(self, word: str) -> list[tuple[int, str]]:
        """Token pieces spelling `word` (identity-parity injection)."""
        try:
            ids = self._llm.tokenize(word.encode("utf-8"), add_bos=False, special=False)
            return [(i, self.detokenize_text(i) or "") for i in ids]
        except Exception:
            return []

    def detokenize_text(self, token_id: int) -> str | None:
        """Plain text of one token id (injection probes)."""
        try:
            return self._llm.detokenize([token_id], special=False).decode("utf-8", errors="replace")
        except Exception:
            return None

    def save_state(self) -> bytes:
        # an opaque in-memory handle: (llama_cpp.LlamaState, last logits);
        # the context buffer alone does not carry the scored distribution.
        return (self._llm.save_state(), self._last_logits)

    def load_state(self, state: bytes) -> None:
        lm_state, logits = state
        self._llm.load_state(lm_state)
        self._last_logits = logits


def _common_suffix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return a[len(a) - n:]
