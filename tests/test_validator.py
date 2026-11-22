"""The validation agent's plumbing: evidence, the checker, the rules baseline and the loop."""

import json

import pytest

from creditadj import validator as V


@pytest.fixture(scope="module")
def candidates(art):
    from creditadj.candidates import build_candidates

    return {c.id: c for c in build_candidates()}


@pytest.fixture(scope="module")
def packs(candidates):
    from creditadj.validation import evidence_pack

    return {cid: evidence_pack(c) for cid, c in candidates.items()}


def seen_of(pack):
    return {i["id"]: i["text"] for result in pack.values() for i in result["items"]}


# ---- number policy ----


def test_numbers_are_read_with_their_decimals():
    found = V.extract_numbers("AUC 0.768 on 6,000 accounts, +0.016 against 22.1%, util_avg_6m, SR 11-7, E-23, PERF-GAP.")
    assert [(n.text, n.decimals) for n in found] == [("0.768", 3), ("6,000", 0), ("+0.016", 3), ("22.1", 1)]


@pytest.mark.parametrize(
    "claim, evidence, ok",
    [
        ("0.768", "test AUC 0.768", True),
        ("0.77", "test AUC 0.768", True),  # rounding to fewer decimals is allowed
        ("0.769", "test AUC 0.768", False),
        ("0.7680", "test AUC 0.768", False),  # more decimals than the evidence has
        ("77%", "test AUC 0.768", False),  # no unit conversion
        ("45.3", "predicted 45.3%", True),
        ("0.016", "difference +0.016", True),  # sign is ignored
        ("3,000", "3,000 of 6,000", True),
        ("3000", "3,000 of 6,000", True),
        ("2,999", "3,000 of 6,000", False),
    ],
)
def test_rounding_policy(claim, evidence, ok):
    number = V.extract_numbers(claim)[0]
    assert V.supported(number, V.extract_numbers(evidence)) is ok


def test_small_whole_numbers_are_exempt_but_percentages_of_any_size_are_checked_as_numbers():
    assert V.is_exempt(V.extract_numbers("7 inputs")[0])
    assert not V.is_exempt(V.extract_numbers("12 inputs")[0])
    assert not V.is_exempt(V.extract_numbers("0.5")[0])


# ---- the checker ----


def memo(decision, *findings, summary="The model has a serious problem."):
    return {"decision": decision, "summary": summary, "findings": [dict(zip(("severity", "title", "text", "evidence"), f)) for f in findings]}


def test_checker_accepts_a_grounded_memo(packs):
    seen = seen_of(packs["C"])
    good = memo("reject", ("high", "Overfitting", "Training AUC exceeds test AUC by 0.265.", ["PERF-GAP"]))
    assert V.check_memo(good, seen).passed


@pytest.mark.parametrize(
    "bad, kind",
    [
        (("high", "Overfitting", "Training AUC exceeds test AUC by 0.300.", ["PERF-GAP"]), "unsupported_number"),
        (("high", "Overfitting", "The gap is large.", []), "no_evidence"),
        (("high", "Overfitting", "The gap is large.", ["PERF-MADEUP"]), "unknown_evidence"),
        (("high", "Overfitting", "The gap is 0.265 and the overlap is 3,001.", ["PERF-GAP"]), "unsupported_number"),
        (("high", "Overfitting", "The gap is 0.265 and test AUC is 0.735.", ["PERF-GAP"]), "number_not_cited"),
    ],
)
def test_checker_refuses_ungrounded_findings(packs, bad, kind):
    check = V.check_memo(memo("reject", bad), seen_of(packs["C"]))
    assert not check.passed and kind in {p["kind"] for p in check.problems}


def test_checker_only_accepts_evidence_the_agent_was_shown(packs):
    finding = ("high", "Overfitting", "Training AUC exceeds test AUC by 0.265.", ["PERF-GAP"])
    assert not V.check_memo(memo("reject", finding), {}).passed


@pytest.mark.parametrize(
    "decision, severity, ok",
    [("approve", "high", False), ("approve", "medium", False), ("approve", "low", True),
     ("approve_with_conditions", "medium", True), ("approve_with_conditions", "low", False),
     ("reject", "high", True), ("reject", "medium", False)],
)
def test_decision_must_follow_from_severities(packs, decision, severity, ok):
    finding = (severity, "Gap", "Training AUC exceeds test AUC by 0.265.", ["PERF-GAP"])
    check = V.check_memo(memo(decision, finding), seen_of(packs["C"]))
    assert ("inconsistent" not in {p["kind"] for p in check.problems}) is ok


# ---- the submissions and the rules baseline ----


def test_every_planted_defect_is_real_and_visible_in_the_evidence(packs):
    text = {cid: seen_of(pack) for cid, pack in packs.items()}
    assert "0 of 6,000" in text["A"]["DATA-OVERLAP"] and "3,000 of 6,000" in text["H"]["DATA-OVERLAP"]
    assert V.extract_numbers(text["C"]["PERF-GAP"])[0].value > 0.2 > V.extract_numbers(text["A"]["PERF-GAP"])[0].value
    assert "collections_flag" in text["B"]["INP-LIST"] and "after the observation window" in text["B"]["FEAT-collections_flag-DICT"]
    assert "age" in text["E"]["INP-LIST"].split(": ")[1] and "INP-age" not in text["A"]
    predicted, observed = (n.value for n in V.extract_numbers(text["D"]["CAL-MEAN"])[:2])
    assert predicted - observed > 15
    assert V.extract_numbers(text["F"]["STAB-pay_ratio"])[-1].value > 0.25 > V.extract_numbers(text["A"]["STAB-SCORE"])[-1].value
    assert "-0.1" in text["G"]["BENCH-DIFF"]


def test_evidence_never_states_a_verdict(packs):
    banned = ("leak", "overfit", "prohibited", "unacceptable", "fail", "defect")
    for cid, pack in packs.items():
        for evidence_id, text in seen_of(pack).items():
            if evidence_id.endswith("-NOTE"):
                continue  # conventions name the usual reading, on purpose
            assert not any(word in text.lower() for word in banned), (cid, evidence_id, text)


def test_rules_baseline_catches_every_planted_defect_and_passes_the_checker(candidates):
    for cid, c in candidates.items():
        review = V.review_with_rules(c)
        assert review["check"]["passed"], (cid, review["check"]["problems"])
        assert review["score"]["decision_ok"], cid
        if cid == "A":
            assert review["score"]["false_alarms"] == 0 and review["memo"]["decision"] != "reject"
        else:
            assert review["score"]["caught"] is True, cid


def test_scoring_needs_the_right_evidence_not_just_a_rejection(candidates):
    planted = candidates["H"].planted
    wrong_reason = memo("reject", ("high", "Overfitting", "The gap is large.", ["PERF-GAP"]))
    right_reason = memo("reject", ("high", "Overlap", "Test rows were used in training.", ["DATA-OVERLAP"]))
    assert V.score_memo(wrong_reason, planted)["caught"] is False
    assert V.score_memo(right_reason, planted)["caught"] is True


# ---- the agent loop, driven by a scripted stand-in for the model ----


def scripted(*turns):
    turns = list(turns)

    def chat(messages, tools):
        assert tools[0]["function"]["name"] == "run_test"
        calls = turns.pop(0)
        return {"content": "", "tool_calls": [{"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]}

    return chat


def test_agent_loop_serves_tests_and_stops_at_the_memo(packs, candidates):
    final = {"decision": "reject", "summary": "Test rows overlap training.", "findings": [
        {"severity": "high", "title": "Overlap", "text": "3,000 of 6,000 test accounts were used in training.", "evidence": ["DATA-OVERLAP"]}]}
    chat = scripted(
        [("run_test", {"name": "performance"}), ("run_test", {"name": "data_integrity"})],
        [("run_test", {"name": "fairness", "argument": "sex"}), ("run_test", {"name": "feature_detail", "argument": "nope"})],
        [("submit_memo", final)],
    )
    run = V.run_agent(packs["H"], "Submission H", "summary", chat)
    assert run.stopped == "memo" and run.turns == 3 and len(run.calls) == 4
    assert "DATA-OVERLAP" in run.seen and "FAIR-sex-AIR" in run.seen and "TEST-ARG" in run.seen
    assert V.check_memo(run.memo, run.seen).passed
    assert V.score_memo(run.memo, candidates["H"].planted)["caught"] is True


def test_agent_loop_enforces_the_test_budget(packs):
    one_turn = [("run_test", {"name": "performance"})] * (V.MAX_TOOL_CALLS + 3)
    run = V.run_agent(packs["A"], "Submission A", "summary", scripted(one_turn, [("submit_memo", {"decision": "approve", "summary": "", "findings": []})]))
    assert len(run.calls) == V.MAX_TOOL_CALLS and run.stopped == "memo"


def test_agent_loop_reports_a_model_that_never_submits(packs):
    run = V.run_agent(packs["A"], "Submission A", "summary", lambda messages, tools: {"content": "I think it is fine."})
    assert run.memo is None and run.stopped == "no_tool"


def test_validation_export_has_what_the_web_demo_reads(packs):
    from creditadj.export import build_validation_payload

    payload = build_validation_payload()
    assert [s["id"] for s in payload["submissions"]] == list("ABCDEFGH")
    assert payload["system_prompt"] == V.SYSTEM_PROMPT and payload["max_tool_calls"] == V.MAX_TOOL_CALLS
    for s in payload["submissions"]:
        assert set(s) == {"id", "label", "developer_summary", "inputs", "pack", "rules", "planted"}
        assert s["planted"]["key"] not in s["developer_summary"] and s["planted"]["title"].lower() not in s["developer_summary"].lower()
        assert "performance" in s["pack"] and f"feature_detail:{s['inputs'][0]}" in s["pack"]


def test_evidence_ids_are_read_off_the_front_of_whatever_the_model_hands_back():
    assert V.clean_evidence_id("PERF-GAP") == "PERF-GAP"
    assert V.clean_evidence_id("[PERF-GAP]") == "PERF-GAP"
    assert V.clean_evidence_id("[FEAT-collections_flag-DICT] Data dictionary entry for collections_flag: 1 if ...") == "FEAT-collections_flag-DICT"
    assert V.clean_evidence_id(" FAIR-age_band-AIR: 0.786") == "FAIR-age_band-AIR"
    assert V.clean_evidence_id("the performance test") == "the performance test"
    memo = V.normalise_memo({"decision": "reject", "summary": "", "findings": [{"severity": "high", "title": "t", "text": "x", "evidence": ["[INP-age] age: 2.1%"]}]})
    assert memo["findings"][0]["evidence"] == ["INP-age"]
