"""The validation agent, the checker that keeps it honest, and the scoring.

Three parts:

* An agent loop. A language model is given the developer's summary of a
  submission and one tool, `run_test`. It decides which tests to run and in
  what order, reads the evidence, and ends by calling `submit_memo` with a
  decision and a list of findings. Any chat-completions endpoint with tool
  calling can drive it (`VALIDATOR_LLM_*` environment variables).

* A deterministic checker. It never asks a model anything. It confirms that
  every finding cites evidence the agent was actually shown, that every number
  in a finding appears in the evidence it cites, and that the decision follows
  from the severities. A number that is missing from the cited evidence is
  reported one of two ways: "number_not_cited" if it is somewhere in what the
  agent was shown (a citation slip), "unsupported_number" if it is nowhere
  (invented, or the result of arithmetic).

* A rules baseline and the scoring. The baseline runs every test and applies
  fixed thresholds. Both the baseline and the agent are scored against the
  hidden answer key in `candidates.py`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Callable

from .candidates import Candidate, Planted
from .validation import FAIRNESS_ATTRIBUTES, TESTS, evidence_pack

MAX_TOOL_CALLS = 14
DECISIONS = ("approve", "approve_with_conditions", "reject")
SEVERITIES = ("high", "medium", "low")

SYSTEM_PROMPT = f"""You are an independent model validator at a bank. A model developer has submitted a credit default model for approval. Your job is to challenge it, not to help it pass.

How to work:
- Use run_test to run validation tests. Each result is a list of evidence items. An item has an id and one sentence.
- Start broad: performance, benchmark, calibration, data_integrity, inputs and stability. Then follow up anything unusual. Look up the data dictionary entry (feature_detail) of any input that dominates the model or that is not a standard payment-behaviour field, and check fairness for more than one attribute.
- You may ask for several tests in one turn. You have at most {MAX_TOOL_CALLS} test runs in total.
- A result that looks too good deserves as much suspicion as one that looks bad.

What a validator checks:
- Conceptual soundness: is every input legitimate, and is it known at the moment the credit decision is made? Age, sex and marital status are prohibited bases for a credit decision.
- Outcomes: discrimination, calibration, rank ordering, and whether the model beats a simple benchmark.
- Data integrity: are the samples independent of each other?
- Ongoing monitoring: has the population the model now scores moved away from the one it was built on?
- Fair-lending outcomes by group.

When you have enough evidence, call submit_memo exactly once.
- severity "high": the model must not be used as submitted. "medium": usable only once a named condition is met. "low": worth recording, no action needed before use.
- decision: "reject" if any finding is high, "approve_with_conditions" if the worst finding is medium, "approve" if every finding is low or there are none.
- Each finding has a short title, one to three sentences of plain prose, and the ids of the evidence items that support it.
- Quote numbers exactly as they appear in the evidence you cite. Never state a number that is not in an evidence item cited by the same finding. Do not convert units or do arithmetic.
- Do not write a finding for a test that came back clean.
- The summary is two or three sentences with no numbers in it.
- Write plainly, with no headings or list marks inside the text."""

TOOLS = [
    {
        "name": "run_test",
        "description": "Run one validation test on the submitted model and return its evidence items. Tests: "
        + " ".join(f"{name}: {desc}" for name, (desc, _) in TESTS.items())
        + f" feature_detail needs argument set to an input name. fairness needs argument set to one of: {', '.join(FAIRNESS_ATTRIBUTES)}.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(TESTS)},
                "argument": {"type": "string", "description": "Input name for feature_detail, or attribute for fairness. Leave out for other tests."},
            },
            "required": ["name"],
        },
    },
    {
        "name": "submit_memo",
        "description": "Submit the validation memo. Call this once, when the evidence is in. It ends the review.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(DECISIONS)},
                "summary": {"type": "string", "description": "Two or three plain sentences, no numbers."},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string", "enum": list(SEVERITIES)},
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Evidence ids only, for example PERF-GAP. Not the sentence that follows the id.",
                            },
                        },
                        "required": ["severity", "title", "text", "evidence"],
                    },
                },
            },
            "required": ["decision", "summary", "findings"],
        },
    },
]


_LEADING_ID = re.compile(r"\s*\[?([A-Z]{3,5}-[A-Za-z0-9_]+(?:-[A-Z]+)?)\]?")


def clean_evidence_id(value: object) -> str:
    """Accept an id with or without brackets, and ignore any text after it.

    Tool results are shown to the model as "[PERF-GAP] sentence", and models
    often hand the whole line back. That is a formatting habit, not a wrong
    citation, so the harness reads the id off the front.
    """
    text = str(value)
    match = _LEADING_ID.match(text)
    return match.group(1) if match else text.strip()


def normalise_memo(memo: dict) -> dict:
    """Tidy a submitted memo before it is checked: evidence entries become bare ids."""
    out = dict(memo)
    if isinstance(out.get("findings"), list):
        out["findings"] = [
            {**f, "evidence": [clean_evidence_id(e) for e in (f.get("evidence") or [])]} if isinstance(f, dict) else f
            for f in out["findings"]
        ]
    return out


def first_message(candidate_label: str, developer_summary: str) -> str:
    return f"{candidate_label}. The developer's summary: {developer_summary}\n\nValidate this model."


def pack_key(name: str, argument: str | None) -> str:
    return f"{name}:{argument}" if TESTS.get(name, ("", None))[1] and argument else name


def lookup(pack: dict, name: str, argument: str | None) -> dict:
    """Serve one tool call from a precomputed evidence pack."""
    key = pack_key(name, (argument or "").strip() or None)
    if key in pack:
        return pack[key]
    if name not in TESTS:
        return {"test": name, "items": [{"id": "TEST-NONE", "text": f"There is no test called {name}."}]}
    needs = TESTS[name][1]
    return {"test": name, "items": [{"id": "TEST-ARG", "text": f"{name} needs a valid {needs}. '{argument or ''}' is not one."}]}


# --------------------------------------------------------------------------
# The checker
# --------------------------------------------------------------------------

# Names of regulations look like numbers and are not claims about the model.
_NOT_NUMBERS = re.compile(r"\bSR\s?\d{1,2}-\d{1,2}\b|\bE-23\b|\bOCC\s?\d{4}-\d{1,2}\b|\bSS\d/\d{2}\b", re.IGNORECASE)
_EVIDENCE_ID = re.compile(r"\b[A-Z]{3,5}-[A-Za-z0-9_]+(?:-[A-Z]+)?\b")
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w])(?!\.\d)")


def _digits(text: str) -> str:
    return text.replace(",", "").lstrip("+-")


@dataclass(frozen=True)
class Number:
    text: str
    value: float
    decimals: int


def extract_numbers(text: str) -> list[Number]:
    """Every number written in digits, with the count of decimals it was written to."""
    clean = _EVIDENCE_ID.sub(" ", _NOT_NUMBERS.sub(" ", text))
    out = []
    for match in _NUMBER.finditer(clean):
        raw = match.group(0)
        digits = raw.replace(",", "").lstrip("+-")
        if not digits or digits == ".":
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        out.append(Number(raw, abs(float(digits)), decimals))
    return out


def is_exempt(number: Number) -> bool:
    """Whole numbers from 0 to 10 are not checked: they are usually counts in the memo's own prose."""
    return number.decimals == 0 and number.value <= 10


def supported(number: Number, evidence: list[Number]) -> bool:
    """True if the number equals an evidence number, or that number rounded to fewer decimals.

    The sign and a percent sign are ignored. Nothing else is: 0.786 does not
    support "79%", and no arithmetic on evidence is accepted.
    """
    target = Decimal(_digits(number.text))
    step = Decimal(1).scaleb(-number.decimals)
    for item in evidence:
        if item.decimals < number.decimals:
            continue
        # Exact decimal arithmetic, ties to even, so the browser port can match it digit for digit.
        if Decimal(_digits(item.text)).quantize(step, rounding=ROUND_HALF_EVEN) == target:
            return True
    return False


@dataclass
class Check:
    passed: bool
    problems: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "problems": self.problems}


def check_memo(memo: dict, seen: dict[str, str]) -> Check:
    """Check a memo against the evidence the agent was shown (`seen`: id -> text)."""
    problems: list[dict] = []

    def add(where: str, kind: str, detail: str) -> None:
        problems.append({"where": where, "kind": kind, "detail": detail})

    decision = memo.get("decision")
    findings = memo.get("findings")
    if decision not in DECISIONS:
        add("decision", "bad_decision", f"decision must be one of {', '.join(DECISIONS)}")
    if not isinstance(findings, list):
        add("findings", "bad_findings", "findings must be a list")
        findings = []

    shown = [n for text in seen.values() for n in extract_numbers(text)]
    cited_everywhere: list[Number] = []
    for i, finding in enumerate(findings):
        where = f"finding {i + 1}"
        severity = finding.get("severity")
        if severity not in SEVERITIES:
            add(where, "bad_severity", f"severity must be one of {', '.join(SEVERITIES)}")
        ids = [e for e in finding.get("evidence") or [] if isinstance(e, str)]
        if not ids:
            add(where, "no_evidence", "the finding cites no evidence")
        known = [e for e in ids if e in seen]
        for e in ids:
            if e not in seen:
                add(where, "unknown_evidence", f"{e} was not returned by any test in this review")
        evidence_numbers = [n for e in known for n in extract_numbers(seen[e])]
        cited_everywhere += evidence_numbers
        for number in extract_numbers(f"{finding.get('title', '')}. {finding.get('text', '')}"):
            if is_exempt(number) or supported(number, evidence_numbers):
                continue
            if supported(number, shown):
                # Usually a citation slip: the figure is real, the finding just does not cite its source.
                add(where, "number_not_cited", f"{number.text} is in evidence the agent was shown, but not in what this finding cites")
            else:
                # Not in anything the agent was shown: invented, or worked out by arithmetic.
                add(where, "unsupported_number", f"{number.text} is not in any evidence the agent was shown")

    for number in extract_numbers(str(memo.get("summary", ""))):
        if is_exempt(number) or supported(number, cited_everywhere):
            continue
        kind = "number_not_cited" if supported(number, shown) else "unsupported_number"
        add("summary", kind, f"{number.text} is not in any evidence the memo cites")

    severities = [f.get("severity") for f in findings]
    if decision == "approve" and ("high" in severities or "medium" in severities):
        add("decision", "inconsistent", "approve is not available when a finding is high or medium")
    if decision == "approve_with_conditions" and ("high" in severities or "medium" not in severities):
        add("decision", "inconsistent", "approve_with_conditions needs a medium finding and no high finding")
    if decision == "reject" and "high" not in severities:
        add("decision", "inconsistent", "reject needs at least one high finding")
    return Check(passed=not problems, problems=problems)


# --------------------------------------------------------------------------
# Scoring against the answer key
# --------------------------------------------------------------------------


def score_memo(memo: dict, planted: Planted | dict) -> dict:
    """Did the memo catch what was planted, and is the decision defensible?"""
    key = planted.key if isinstance(planted, Planted) else planted["key"]
    prefixes = tuple(planted.evidence_prefixes if isinstance(planted, Planted) else planted["evidence_prefixes"])
    findings = memo.get("findings") or []

    def hits(finding: dict) -> bool:
        return any(str(e).startswith(prefixes) for e in finding.get("evidence") or []) if prefixes else False

    serious = [f for f in findings if f.get("severity") in ("high", "medium")]
    decision = memo.get("decision")
    if key == "none":
        return {
            "planted": key,
            "caught": None,
            "decision_ok": decision != "reject",
            "false_alarms": sum(1 for f in findings if f.get("severity") == "high"),
        }
    return {
        "planted": key,
        "caught": any(hits(f) for f in serious),
        "decision_ok": decision in ("reject", "approve_with_conditions"),
        "false_alarms": 0,
    }


# --------------------------------------------------------------------------
# Rules baseline
# --------------------------------------------------------------------------

_DEMOGRAPHIC = ("age", "sex_female", "married", "single", "edu_graduate", "edu_university", "edu_high_school")


def _first_number(text: str, after: str = "") -> float:
    tail = text[text.index(after) + len(after):] if after else text
    return extract_numbers(tail)[0].value


def rules_memo(pack: dict) -> dict:
    """Run every test and apply fixed thresholds. No model is involved.

    The thresholds were written by the same person who planted the defects,
    so this baseline is expected to catch them. It is here to show what the
    evidence supports and to give the agent something to be compared with.
    """
    item = {i["id"]: i["text"] for result in pack.values() for i in result["items"]}
    findings: list[dict] = []

    def add(severity: str, title: str, text: str, evidence: list[str]) -> None:
        findings.append({"severity": severity, "title": title, "text": text, "evidence": evidence})

    gap = _first_number(item["PERF-GAP"], ":")
    if gap > 0.05:
        add("high", "Overfitting", f"Training AUC exceeds test AUC by {gap:.3f}, above the usual limit of 0.05.", ["PERF-GAP", "PERF-NOTE"])

    diff_text = item["BENCH-DIFF"]
    diff = float(re.search(r"[+-]\d\.\d{3}", diff_text).group(0))
    if diff < -0.01:
        add("high", "Worse than the benchmark", f"The submission's test AUC differs from a plain logistic regression by {diff:+.3f}.", ["BENCH-DIFF"])

    predicted, observed = (n.value for n in extract_numbers(item["CAL-MEAN"])[:2])
    if abs(predicted - observed) > 5:
        add("high", "Miscalibrated", f"Mean predicted default is {predicted:.1f}% against {observed:.1f}% observed.", ["CAL-MEAN"])

    overlap = extract_numbers(item["DATA-OVERLAP"])[0]
    if overlap.value > 0:
        add("high", "Test sample overlaps training", f"{overlap.text} test accounts were also used in training, so the test results are not independent.", ["DATA-OVERLAP"])

    for key in _DEMOGRAPHIC:
        if f"INP-{key}" in item:
            add("high", "Prohibited input", f"{key} is a model input. Demographic fields are not a permitted basis for a credit decision.", [f"INP-{key}"])
            break

    for evidence_id, text in item.items():
        if evidence_id.startswith("INP-") and evidence_id != "INP-LIST":
            single = extract_numbers(text)[-1]
            if single.value > 0.85:
                name = evidence_id[4:]
                add("high", "Possible target leakage", f"{name} reaches an AUC of {single.text} on its own, which is not plausible for a field known before the outcome.", [evidence_id, f"FEAT-{name}-DICT"])

    feature_psi = [(k, extract_numbers(t)[-1]) for k, t in item.items() if k.startswith("STAB-") and k not in ("STAB-SCORE", "STAB-MEAN", "STAB-NOTE")]
    score_psi = extract_numbers(item["STAB-SCORE"])[-1]
    moved = [(k, n) for k, n in feature_psi if n.value >= 0.25]
    if score_psi.value >= 0.25 or moved:
        cited = ["STAB-SCORE"] + [k for k, _ in moved] + ["STAB-NOTE"]
        detail = ", ".join(f"{k[5:]} at {n.text}" for k, n in moved) or f"the score at {score_psi.text}"
        add("high" if score_psi.value >= 0.25 else "medium", "Population has moved", f"Population stability index: {detail}. The convention reads 0.25 or more as a material shift.", cited)

    for attribute in FAIRNESS_ATTRIBUTES:
        text = item.get(f"FAIR-{attribute}-AIR", "")
        numbers = extract_numbers(text)
        if numbers and numbers[-1].value < 0.80:
            add("medium", f"Approval gap by {attribute.replace('_', ' ')}", f"The adverse-impact ratio is {numbers[-1].text}, under the 0.80 screen.", [f"FAIR-{attribute}-AIR", "FAIR-NOTE"])

    if "do not rise" in item["RANK-INV"]:
        add("low", "Rank ordering is not strict", "Default rates do not rise in every score decile.", ["RANK-INV"])
    add("low", "No out-of-time test", "The data is one snapshot with a random split, so performance on a later period is unmeasured.", ["DATA-WINDOW"])

    severities = {f["severity"] for f in findings}
    decision = "reject" if "high" in severities else "approve_with_conditions" if "medium" in severities else "approve"
    return {"decision": decision, "summary": "Produced by fixed thresholds applied to every test. No language model was used.", "findings": findings}


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------

ChatFn = Callable[[list[dict], list[dict]], dict]


@dataclass
class Run:
    memo: dict | None
    calls: list[dict]  # {"name", "argument", "ids"}
    seen: dict[str, str]
    turns: int
    stopped: str  # "memo", "budget", "no_tool", "error"

    def to_dict(self) -> dict:
        return {"memo": self.memo, "calls": self.calls, "turns": self.turns, "stopped": self.stopped}


def run_agent(pack: dict, label: str, developer_summary: str, chat: ChatFn, max_turns: int = 10) -> Run:
    """Drive one review. `chat(messages, tools)` returns a chat-completions message dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_message(label, developer_summary)},
    ]
    tools = [{"type": "function", "function": t} for t in TOOLS]
    calls: list[dict] = []
    seen: dict[str, str] = {}

    for turn in range(1, max_turns + 1):
        reply = chat(messages, tools)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            return Run(None, calls, seen, turn, "no_tool")
        messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": tool_calls})
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "submit_memo":
                return Run(normalise_memo(args), calls, seen, turn, "memo")
            if len(calls) >= MAX_TOOL_CALLS:
                content = "No test runs left. Call submit_memo now."
            else:
                result = lookup(pack, str(args.get("name", "")), args.get("argument"))
                for item in result["items"]:
                    seen[item["id"]] = item["text"]
                calls.append({"name": args.get("name"), "argument": args.get("argument"), "ids": [i["id"] for i in result["items"]]})
                content = "\n".join(f"[{i['id']}] {i['text']}" for i in result["items"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    return Run(None, calls, seen, max_turns, "budget")


def chat_from_env(env: dict | None = None) -> ChatFn | None:
    """A chat function for any chat-completions endpoint, or None if no key is set."""
    env = os.environ if env is None else env
    key = env.get("VALIDATOR_LLM_API_KEY", "").strip()
    if not key:
        return None
    base = env.get("VALIDATOR_LLM_BASE_URL", "").strip().rstrip("/")
    name = env.get("VALIDATOR_LLM_MODEL", "").strip()
    if not base or not name:
        raise RuntimeError("VALIDATOR_LLM_API_KEY is set, so VALIDATOR_LLM_BASE_URL and VALIDATOR_LLM_MODEL must be set as well")

    def chat(messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps({"model": name, "messages": messages, "tools": tools, "temperature": 0}).encode()
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode())["choices"][0]["message"]
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"chat-completions request failed: {exc}") from exc

    return chat


def review_with_rules(candidate: Candidate) -> dict:
    """The rules baseline on one submission: memo, check and score."""
    pack = evidence_pack(candidate)
    memo = rules_memo(pack)
    seen = {i["id"]: i["text"] for result in pack.values() for i in result["items"]}
    return {"memo": memo, "check": check_memo(memo, seen).to_dict(), "score": score_memo(memo, candidate.planted)}
