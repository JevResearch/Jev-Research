"""Fresh-task generator v3 tests (LEAD-GATE requirement B).

Includes the LEAD-GATE fault-injection checks: the design-quality validator
must FAIL on seed-ignoring generators, all-absent missing-info answers, fixed
gold positions, graph answers readable from option text, and dev/test
signature collisions.
"""

from __future__ import annotations

import json

import pytest

from jev_observatory.datasets import fresh_v3_verify as verifier
from jev_observatory.datasets.fresh_v3 import (
    DATE_GAPS,
    DAYS_IN_MONTH,
    FreshSpecError,
    fresh_v3_spec,
    make_dates_item,
    make_graph_item,
    make_missing_info_item,
    seed_sensitivity_report,
    substantive_signature,
    validate_fresh_spec,
    _family_of,
)
from jev_observatory.experiment import logical_requests


@pytest.fixture(scope="module")
def spec():
    return fresh_v3_spec()


@pytest.fixture(scope="module")
def items(spec):
    return spec["items"]


# ------------------------------------------------------------------ labels
def test_all_labels_verified_independently(items):
    report = verifier.verify_all(items)
    assert report["n_failures"] == 0
    assert report["all_labels_verified"]


def test_graph_structure_roles(items):
    from jev_observatory.datasets.fresh_v3_verify import bfs_distances, parse_edges

    for item in (i for i in items if i["group"] == "fresh_v3_graph"):
        gold = item["gold"]
        distances = bfs_distances(parse_edges(item["state"]), gold["target_node"])
        assert distances[gold["answer"]["value"]] == 2
        assert distances[gold["role_of"]["distractor"]] == 1
        assert distances.get(gold["role_of"]["negative"], 10 ** 9) > 2
        # no Executive/Manager style role cues anywhere
        text = (item["state"] + json.dumps(item["questions"])).lower()
        assert "executive" not in text and "manager" not in text


def test_dates_both_methods_agree_and_cover_both_outcomes(items):
    dates = [i for i in items if i["group"] == "fresh_v3_dates"]
    same = sum(1 for i in dates if i["gold"]["answer"]["value"] is True)
    cross = sum(1 for i in dates if i["gold"]["answer"]["value"] is False)
    assert same > 0 and cross > 0
    for item in dates:
        start, gap = item["gold"]["start_day"], item["gold"]["gap_days"]
        assert verifier.same_month_by_arithmetic(start, gap) == verifier.same_month_by_stepping(start, gap)


def test_dates_impossible_cells_not_asserted():
    from jev_observatory.datasets.fresh_v3 import dates_coverage_table

    spec = fresh_v3_spec()
    dates = [i for i in spec["items"] if i["group"] == "fresh_v3_dates"]
    table = dates_coverage_table(dates)
    for gap in (30, 31, 45, 50):
        cell = table["cells"][f"gap={gap}"]
        assert cell["same_month_mathematically_possible"] is False
        assert cell["same_month"] == 0
    assert table["impossible_cells"]  # impossibility is documented, not hidden


def test_missing_info_balance_and_disjoint_values(items):
    missing = [i for i in items if i["group"] == "fresh_v3_missing_info"]
    answerable = [i for i in missing if i["gold"]["answerable"]]
    absent = [i for i in missing if not i["gold"]["answerable"]]
    assert abs(len(answerable) - len(absent)) <= 2
    for item in missing:
        criteria = item["questions"]["answer"]["criteria"]
        contents = [v for v in criteria.values() if v != "the registry does not state this"]
        assert len(set(contents)) == len(contents)  # disjoint values
    # not_stated must be correct sometimes and wrong sometimes
    assert 0 < len(absent) < len(missing)


# ------------------------------------------------------- dev/test integrity
def test_dev_test_substantive_signatures_disjoint(items):
    dev = {substantive_signature(i, family=_family_of(i)) for i in items if "-dev-" in i["id"]}
    test = {substantive_signature(i, family=_family_of(i)) for i in items if "-test-" in i["id"]}
    assert len(dev) == 24 and len(test) == 120
    assert not (dev & test)


def test_counts_and_seeds_recorded(spec):
    assert len(spec["items"]) == 144
    groups = {}
    for item in spec["items"]:
        groups[item["group"]] = groups.get(item["group"], 0) + 1
    assert groups == {"fresh_v3_graph": 48, "fresh_v3_missing_info": 48, "fresh_v3_dates": 48}
    assert spec["generator_version"].startswith("fresh_v3-")
    assert spec["seeds"]["per_family"] and spec["seeds"]["test_seed"] != spec["seeds"]["dev_seed"]


def test_seed_sensitivity_probe():
    for family in ("graph_two_hop", "missing_info", "synthetic_dates"):
        report = seed_sensitivity_report(family, seeds=[1, 2, 3], n=10)
        assert report["distinct_signatures_total"] >= 25, family
        assert not report["rename_only_suspected"]


# ------------------------------------------------------- fault injections
def _signature_set(spec_items, split):
    return {substantive_signature(i, family=_family_of(i)) for i in spec_items if f"-{split}-" in i["id"]}


def test_fault_seed_ignored_is_caught():
    """A generator that ignores its seed collapses to rename-only variation."""
    spec = fresh_v3_spec(dev_n=24, test_n=24)
    graph_items = [i for i in spec["items"] if i["group"] == "fresh_v3_graph"]
    base = json.loads(json.dumps(graph_items[0]))
    for k, item in enumerate(graph_items):
        item["state"] = base["state"]
        item["questions"] = json.loads(json.dumps(base["questions"]))
        item["gold"] = json.loads(json.dumps(base["gold"]))
        # keep gold positions balanced so ONLY the rename-only check fires
        criteria = item["questions"]["answer"]["criteria"]
        keys = list(criteria)
        rot = k % len(keys)
        item["questions"]["answer"]["criteria"] = {
            key: criteria[key] for key in keys[rot:] + keys[:rot]
        }
    with pytest.raises(FreshSpecError, match="rename-only"):
        validate_fresh_spec(spec)


def test_fault_all_answers_absent_is_caught():
    spec = fresh_v3_spec(dev_n=12, test_n=12)
    for item in spec["items"]:
        if item["group"] == "fresh_v3_missing_info":
            item["gold"]["answerable"] = False
    with pytest.raises(FreshSpecError, match="answerable share"):
        validate_fresh_spec(spec)


def test_fault_gold_position_fixed_is_caught():
    spec = fresh_v3_spec(dev_n=12, test_n=12)
    for item in spec["items"]:
        if item["questions"]["answer"]["type"] != "choice":
            continue
        criteria = item["questions"]["answer"]["criteria"]
        gold_key = item["gold"]["answer"]["value"]
        if list(criteria)[0] != gold_key:
            item["gold"]["answer"]["value"] = list(criteria)[0]
            if "option_values" in item["gold"]:
                item["gold"]["option_values"][list(criteria)[0]] = item["gold"]["option_values"][gold_key]
    with pytest.raises(FreshSpecError, match="gold position"):
        validate_fresh_spec(spec)


def test_fault_graph_answer_from_role_names_is_caught():
    """A descriptive leak ('exactly two edges away') must be refused."""
    item = make_graph_item(seed=5, item_id="leak-probe")
    gold_key = item["gold"]["answer"]["value"]
    criteria = item["questions"]["answer"]["criteria"]
    criteria[gold_key] = "the person exactly two edges away"
    with pytest.raises(verifier.VerificationError, match="role token"):
        verifier.verify_graph_item(item)


def test_fault_dev_test_signature_overlap_is_caught():
    spec = fresh_v3_spec(dev_n=12, test_n=12)
    # force a dev item to structurally duplicate a test item (content AND gold,
    # so independent verification still passes and only the signature check fires)
    dev_graph = next(i for i in spec["items"] if "-dev-graph-" in i["id"])
    test_graph = next(i for i in spec["items"] if "-test-graph-" in i["id"])
    dev_graph["state"] = test_graph["state"]
    dev_graph["questions"] = json.loads(json.dumps(test_graph["questions"]))
    dev_graph["gold"] = json.loads(json.dumps(test_graph["gold"]))
    with pytest.raises(FreshSpecError, match="overlap"):
        validate_fresh_spec(spec)


# ------------------------------------------------------- schema compatibility
def test_specs_pass_logical_request_expansion(spec):
    requests = logical_requests(spec, shuffle=False)
    assert len(requests) == 144
    for request in requests:
        payload = request.request.to_payload()
        assert set(payload) == {"state", "model", "questions"}


# ------------------------------------------------------- outbound mutations (REVIEW-1 #3)
# Every mutation below alters ONLY outbound semantic content while the gold
# metadata is left unchanged; the independent verifier MUST then fail. No test
# relies on a fabricated secondary copy of the option content.
import re as _re


def _rename_all(item, mapping):
    """Consistently rename entities in state, instructions, criteria keys and gold.

    Replacement names are purely alphabetic (the generator's name domain) and a
    placeholder pass prevents new names colliding with old ones mid-substitution.
    """
    from jev_observatory.datasets.fresh_v3 import NAME_POOL
    text_all = json.dumps(item)
    replacements = [n for n in sorted(NAME_POOL) if not _re.search(rf"\b{n}\b", text_all)]
    assert len(replacements) >= len(mapping)
    two_step = {old: f"\x00{i}\x00" for i, old in enumerate(mapping)}
    final = {f"\x00{i}\x00": replacements[i] for i, old in enumerate(mapping)}

    def sub(text):
        for old, new in two_step.items():
            text = _re.sub(rf"\b{old}\b", new, text)
        for old, new in final.items():
            text = text.replace(old, new)
        return text

    item = json.loads(json.dumps(item))
    item["state"] = sub(item["state"])
    q = item["questions"]["answer"]
    q["instructions"] = sub(q["instructions"])
    if isinstance(q.get("criteria"), dict):
        q["criteria"] = {sub(k): sub(v) for k, v in q["criteria"].items()}
    gold = item["gold"]
    gold["answer"]["value"] = sub(gold["answer"]["value"])
    for k, v in list(gold.items()):
        if isinstance(v, dict):
            gold[k] = {sub(kk): sub(vv) for kk, vv in v.items()}
        elif isinstance(v, str) and v in mapping:
            gold[k] = sub(v)
    return item


def test_mutation_dates_state_gap_changed_verifier_fails():
    item = make_dates_item(seed=7, item_id="mut-dates", start_day=25, gap=5)
    assert verifier.verify_dates_item(item)["ok"]
    # REVIEW-1 repro: change the state's gap prose; gold metadata stays '5'
    item["state"] = item["state"].replace("exactly 5 days", "exactly 50 days") \
        .replace("exactly 5 Days", "exactly 50 Days")
    if "exactly 50" not in item["state"]:
        item["state"] = _re.sub(r"(\b|[^0-9])5(\s*days)", r"\g<1>50\g<2>", item["state"], count=1)
    with pytest.raises(verifier.VerificationError):
        verifier.verify_dates_item(item)


def test_mutation_dates_state_start_changed_verifier_fails():
    item = make_dates_item(seed=7, item_id="mut-dates2", start_day=25, gap=5)
    item["state"] = _re.sub(r"day 25 of Month 1", "day 4 of Month 1", item["state"])
    item["state"] = _re.sub(r"Month 1, day 25", "Month 1, day 4", item["state"])
    with pytest.raises(verifier.VerificationError):
        verifier.verify_dates_item(item)


def test_mutation_dates_gold_label_changed_verifier_fails():
    item = make_dates_item(seed=7, item_id="mut-dates3", start_day=25, gap=5)
    item["gold"]["answer"]["value"] = not item["gold"]["answer"]["value"]
    with pytest.raises(verifier.VerificationError):
        verifier.verify_dates_item(item)


def _one_missing_info_answerable():
    from jev_observatory.datasets.fresh_v3 import make_missing_info_item
    for seed in range(50):
        item = make_missing_info_item(seed=seed, item_id="mi", answerable=True)
        if list(item["questions"]["answer"]["criteria"]).index(item["gold"]["answer"]["value"]) == 0:
            return item
    raise AssertionError("no answerable item with gold at position 0 found")


def test_mutation_missing_info_criteria_gold_value_changed_verifier_fails():
    item = _one_missing_info_answerable()
    assert verifier.verify_missing_info_item(item)["ok"]
    gold_key = item["gold"]["answer"]["value"]
    # REVIEW-1 repro: criteria[goldkey] -> 'WRONG VALUE'; gold.option_values untouched
    item["questions"]["answer"]["criteria"][gold_key] = "WRONG VALUE"
    with pytest.raises(verifier.VerificationError):
        verifier.verify_missing_info_item(item)


def test_mutation_missing_info_target_in_instructions_changed_verifier_fails():
    item = _one_missing_info_answerable()
    target = item["gold"]["target_entity"]
    other = next(e for e in _re.findall(r"- ([A-Za-z]+):", item["state"]) if e != target)
    item["questions"]["answer"]["instructions"] = item["questions"]["answer"]["instructions"].replace(
        f"of {target}?", f"of {other}?")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_missing_info_item(item)


def test_mutation_missing_info_state_value_changed_verifier_fails():
    item = _one_missing_info_answerable()
    value = item["gold"]["option_values"][item["gold"]["answer"]["value"]]
    item["state"] = item["state"].replace(value, "floor 9")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_missing_info_item(item)


def test_mutation_graph_target_changed_verifier_fails():
    from jev_observatory.datasets.fresh_v3 import make_graph_item
    item = make_graph_item(seed=11, item_id="mut-graph")
    assert verifier.verify_graph_item(item)["ok"]
    target = item["gold"]["target_node"]
    other = next(n for n in _re.findall(r"[A-ZÀ-Ža-z]+", item["state"]) if n != target)
    item["questions"]["answer"]["instructions"] = item["questions"]["answer"]["instructions"].replace(
        f"away from {target}?", f"away from {other}?")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_graph_item(item)


def test_mutation_graph_state_edge_changed_verifier_fails():
    from jev_observatory.datasets.fresh_v3 import make_graph_item
    item = make_graph_item(seed=11, item_id="mut-graph2")
    # remove the edge that makes the gold node exactly two hops away
    distractor, positive = item["gold"]["role_of"]["distractor"], item["gold"]["role_of"]["positive"]
    a, b = sorted((distractor, positive))
    item["state"] = item["state"].replace(f"{a}–{b}", f"{a}–Zedq").replace(f"{b}–{a}", f"{b}–Zedq")
    assert f"{a}–{b}" not in item["state"] and f"{b}–{a}" not in item["state"]
    with pytest.raises(verifier.VerificationError):
        verifier.verify_graph_item(item)


# ------------------------------------------------------- signature invariance (REVIEW-1 #5)
def _mapped_names(item):
    """Entity names only: registry line entities plus the gold target, never
    value tokens (e.g. equipment model names) that must stay untouched."""
    entities = set(_re.findall(r"^- ([A-ZÀ-Ž][A-Za-z]+):", item["state"], flags=_re.M))
    target = item["gold"].get("target_entity")
    if target:
        entities.add(target)
    values_text = json.dumps(item["questions"]["answer"]["criteria"])
    entities = {e for e in entities if e not in values_text}
    return {name: f"Witness{i:02d}" for i, name in enumerate(sorted(entities))}


def test_signature_invariant_under_pure_rename():
    from jev_observatory.datasets.fresh_v3 import make_graph_item, make_missing_info_item
    graph = make_graph_item(seed=13, item_id="sig-graph")
    mapping = _mapped_names(graph)
    renamed = _rename_all(graph, mapping)
    assert verifier.verify_graph_item(renamed)["ok"]
    assert substantive_signature(renamed, family="graph_two_hop") == \
        substantive_signature(graph, family="graph_two_hop")
    missing = _one_missing_info_answerable()
    mapping2 = _mapped_names(missing)
    renamed2 = _rename_all(missing, mapping2)
    assert verifier.verify_missing_info_item(renamed2)["ok"]
    assert substantive_signature(renamed2, family="missing_info") == \
        substantive_signature(missing, family="missing_info")


def test_signature_invariant_under_presentation_only_changes():
    from jev_observatory.datasets.fresh_v3 import make_graph_item, make_missing_info_item, make_dates_item
    graph = make_graph_item(seed=17, item_id="sig-pres")
    shuffled = json.loads(json.dumps(graph))
    keys = list(shuffled["questions"]["answer"]["criteria"])
    shuffled["questions"]["answer"]["criteria"] = {
        k: shuffled["questions"]["answer"]["criteria"][k] for k in reversed(keys)}
    assert substantive_signature(shuffled, family="graph_two_hop") == \
        substantive_signature(graph, family="graph_two_hop")

    missing = _one_missing_info_answerable()
    reordered = json.loads(json.dumps(missing))
    lines = reordered["state"].splitlines()
    reordered["state"] = "\n".join([lines[0], *reversed(lines[1:])])
    ckeys = list(reordered["questions"]["answer"]["criteria"])
    reordered["questions"]["answer"]["criteria"] = {
        k: reordered["questions"]["answer"]["criteria"][k] for k in reversed(ckeys)}
    assert verifier.verify_missing_info_item(reordered)["ok"]
    assert substantive_signature(reordered, family="missing_info") == \
        substantive_signature(missing, family="missing_info")

    dates = make_dates_item(seed=19, item_id="sig-dates", start_day=25, gap=5)
    rephrased = json.loads(json.dumps(dates))
    # same start/gap in a different calendar phrasing: signature must not move
    rephrased["state"] = _re.sub(
        r"A review meeting is scheduled for day (\d+) of Month 1\.",
        r"Month 1, day \1: review meeting.", dates["state"])
    assert verifier.verify_dates_item(rephrased)["ok"]
    assert substantive_signature(rephrased, family="synthetic_dates") == \
        substantive_signature(dates, family="synthetic_dates")


def test_signature_changes_when_structure_actually_changes():
    from jev_observatory.datasets.fresh_v3 import make_graph_item
    a = make_graph_item(seed=23, item_id="sig-a")
    b = make_graph_item(seed=24, item_id="sig-b")
    assert substantive_signature(a, family="graph_two_hop") != \
        substantive_signature(b, family="graph_two_hop")


def test_signature_deterministic_across_pythonhashseed():
    """The same items must hash identically under different process hash seeds."""
    import subprocess
    import sys

    snippet = (
        "import json,sys;"
        "from jev_observatory.datasets.fresh_v3 import fresh_v3_spec, substantive_signature, _family_of;"
        "items=fresh_v3_spec()['items'];"
        "print(json.dumps(sorted(substantive_signature(i, family=_family_of(i)) for i in items)))"
    )
    outputs = []
    for hash_seed in ("0", "42"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=300,
            env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin",
                 "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]


# ------------------------------------------------------- dates seed feeds cells (REVIEW-1 #4)
def test_dates_seed_changes_quantitative_cells():
    from jev_observatory.datasets.fresh_v3 import fresh_v3_items
    from jev_observatory.datasets.fresh_v3_verify import parse_dates_state

    def cells(seed):
        items = fresh_v3_items(family="synthetic_dates", n=8, seed=seed, split="probe")
        return {parse_dates_state(i["state"]) for i in items}

    c101, c999 = cells(101), cells(999)
    assert c101 != c999, "seed must feed the quantitative (start, gap) sampling"
    assert len(c101 & c999) <= 2
    # both seeds keep same-month and crossing-month coverage
    from jev_observatory.datasets.fresh_v3 import DAYS_IN_MONTH
    for cellset in (c101, c999):
        assert any(s + g <= DAYS_IN_MONTH for s, g in cellset)
        assert any(s + g > DAYS_IN_MONTH for s, g in cellset)


def test_dates_dev_test_cells_disjoint():
    spec = fresh_v3_spec()
    from jev_observatory.datasets.fresh_v3_verify import parse_dates_state
    dev = {parse_dates_state(i["state"]) for i in spec["items"] if "-dev-dates-" in i["id"]}
    test = {parse_dates_state(i["state"]) for i in spec["items"] if "-test-dates-" in i["id"]}
    assert dev and test and not (dev & test)
