"""Space-free merge-rate samples for the architecture probes.

Continuous, whitespace-free text per writing system. Because there is no
whitespace, Jev's apparent template whitespace-normalization (seen in the
tokenizer battery, where newline/tab probes were the worst fits) cannot perturb
these. The server-reported token count per character directly measures how
aggressively the serving vocabulary merges each script -- the sharpest,
most normalization-robust signal available for exact tokenizer identification.

Non-ASCII is built from code points via chr() so this file stays pure ASCII.
"""

from __future__ import annotations

import random



def _u(*codepoints: int) -> str:
    return "".join(chr(c) for c in codepoints)


# name -> (script, whitespace-free sample text)
def _samples() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    # LATIN
    out.append(("latin_words", "latin", "internationalization" * 8))
    out.append(("latin_hello", "latin", "hello" * 30))
    out.append(("latin_repeat_a", "latin", "a" * 200))
    out.append(("latin_random", "latin",
                "".join(random.Random(9).choice("abcdefghijklmnopqrstuvwxyz")
                        for _ in range(200))))
    out.append(("latin_bigram", "latin", "th" * 80))
    # CYRILLIC (Russian) -- the strongest discriminator in the battery
    rv = _u(0x440, 0x443, 0x441, 0x441, 0x43a, 0x438, 0x439)  # руccкий-ish
    out.append(("cyrillic_word", "cyrillic", rv * 20))
    out.append(("cyrillic_common", "cyrillic",
                _u(0x43f, 0x440, 0x438, 0x432, 0x435, 0x442) * 20))  # привет
    out.append(("cyrillic_random", "cyrillic",
                "".join(chr(random.Random(10).randint(0x0410, 0x044f))
                        for _ in range(200))))
    out.append(("cyrillic_sp", "cyrillic", _u(0x441, 0x43e, 0x431, 0x430, 0x43a) * 22))
    # CJK ideographs
    out.append(("cjk_common", "cjk", _u(0x4f60, 0x597d, 0x4e16, 0x754c) * 22))
    out.append(("cjk_random", "cjk",
                "".join(chr(random.Random(11).randint(0x4e00, 0x9fa5))
                        for _ in range(160))))
    out.append(("cjk_numbers", "cjk", _u(0x4e00, 0x4e8c, 0x4e09, 0x56db) * 28))
    # JAPANESE kana
    out.append(("kana_hira", "kana", _u(0x3042, 0x3044, 0x3046, 0x3048, 0x304a) * 22))
    # KOREAN hangul
    out.append(("hangul", "hangul", _u(0xd55c, 0xad6d, 0xc5b4) * 30))
    # GREEK
    out.append(("greek", "greek", _u(0x03b1, 0x03b2, 0x03b3, 0x03b4, 0x03b5) * 24))
    # ARABIC
    out.append(("arabic", "arabic", _u(0x0639, 0x0631, 0x0628, 0x064a, 0x0629) * 24))
    # HEBREW
    out.append(("hebrew", "hebrew", _u(0x05e9, 0x05dc, 0x05d5, 0x05dd) * 30))
    # THAI
    out.append(("thai", "thai", _u(0x0e2a, 0x0e27, 0x0e21, 0x0e2a) * 26))
    # DEVANAGARI
    out.append(("devanagari", "devanagari",
                _u(0x0928, 0x092e, 0x0938, 0x094d, 0x0924, 0x0947) * 20))
    # NUMERICS
    out.append(("digits_pi", "digits",
                "31415926535897932384626433832795028841971" * 5))
    out.append(("digits01", "digits", "01" * 90))
    out.append(("digits_single", "digits", "7" * 120))
    out.append(("digits_years", "digits", "198919901991199219931994" * 8))
    # STRUCTURE-SENSITIVE
    out.append(("urlish", "ascii", "https://example.com/a/b?q=" * 8))
    out.append(("codeish", "code", "def f(x):{return x+1}" * 10))
    out.append(("md_link", "code", "[text](http://url)" * 12))
    out.append(("repeated_pipe", "ascii", "|" * 100))
    out.append(("repeated_dash", "ascii", "-" * 100))
    # MIXED / CODE-SWITCH
    out.append(("codeswitch", "mixed",
                _u(0x440, 0x443) + "hello" + _u(0x4f60, 0x597d) + "world" * 3))
    # EMOJI / RARE PLANE
    out.append(("emoji_run", "emoji", _u(0x1f600) * 24))
    out.append(("flags", "emoji", _u(0x1f1ef, 0x1f1f5) * 8))
    return out


SAMPLES = _samples()
BY_NAME = {name: (script, text) for name, script, text in SAMPLES}


def build_mergerate(*, reps: int = 4, seed: int = 21) -> list[ProbeCall]:
    """Space-free long samples per script; per-char merge-rate oracle.

    reps>=4 so the ~2-tick server noise can be medianed out of each rate.
    """
    from .arch_probe import ProbeCall, _choice  # lazy: avoid circular import

    calls: list[ProbeCall] = []
    for name, script, sample in SAMPLES:
        state = "SEQ:" + sample + ":END"
        for rep in range(reps):
            calls.append(ProbeCall(
                "mergerate", f"mergerate:{name}:r{rep}",
                _choice(["yes", "no"], "Is the sequence empty?", state),
                {"sample": name, "script": script, "rep": rep,
                 "n_chars": len(sample)}))
    return calls


def analyze_mergerate(rows: list[dict]) -> dict:
    """Per-sample median reported input tokens + chars (merge-rate table)."""
    fam = [r for r in rows if r.get("family") == "mergerate"
           and isinstance(r.get("usage_input_tokens"), int)
           and r.get("http_status") == 200]
    by: dict[str, list[dict]] = {}
    for r in fam:
        by.setdefault(r["sample"], []).append(r)
    from statistics import median
    out = {}
    for name, rs in sorted(by.items()):
        nch = rs[0].get("n_chars") or 1
        out[name] = {"script": rs[0].get("script"), "n_chars": nch,
                     "median_reported_in": float(median(r["usage_input_tokens"] for r in rs)),
                     "median_state_chars": float(median(r["state_chars"] for r in rs)),
                     "n_reps": len(rs)}
    return {"family": "mergerate", "claim_type": "exploratory", "samples": out}
