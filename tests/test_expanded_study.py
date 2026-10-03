from jev_observatory.expanded_study import candidate_pool, count_candidates, freeze_plan, select_pairs


def test_pool_has_script_and_control_families():
    rows = candidate_pool()
    families = {x.family for x in rows}
    assert {"latin", "cyrillic", "greek", "cjk", "code", "numeric"} <= families
    assert any(x.variant == "raw" for x in rows)


def test_selection_uses_only_counts_and_is_deterministic():
    rows = count_candidates(candidate_pool(), lambda s: len(s), lambda s: len(s.encode()))
    a = select_pairs(rows)
    b = select_pairs(rows)
    assert a == b
    assert all(x.relation in {"reported-matched", "offline-matched"} for x in a)


def test_freeze_plan_hash_is_stable():
    rows = count_candidates(candidate_pool(), len, lambda s: len(s.encode()))
    pairs = select_pairs(rows)
    assert freeze_plan(pairs) == freeze_plan(pairs)
    assert len(freeze_plan(pairs)["plan_sha256"]) == 64
