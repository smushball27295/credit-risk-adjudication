"""Command line entry point: ``python -m creditadj <command>``.

train        fit the champion and benchmark, print the fit summary
evaluate     discrimination, deciles, calibration, comparison models
cutoff       cutoff table, recommended cutoff, sensitivity to assumptions
explain      decision and reason codes for one test-set row
fairness     approval rates by sex, age band and marital status
monitor      PSI of the test set against train and against a shifted copy
export-web   write web_export/model.json
all          everything above, plus reports/metrics.json and the charts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

from . import SEED, evaluate, export, fairness, model, monitor, plots, reasons
from . import cutoff as cutoff_mod
from .data import default_data_dir
from .features import FEATURE_KEYS, FEATURES
from .pipeline import Artifacts, build, evaluate_all

ROOT = Path(__file__).resolve().parents[1]


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _json_safe(obj):
    """Replace NaN with null so the file is valid JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _build(args) -> Artifacts:
    return build(args.seed, args.lgd, args.margin, args.data_dir)


def cmd_train(args) -> dict:
    art = _build(args)
    tr, va, te = art.split.train, art.split.valid, art.split.test
    y = art.y
    print(f"Rows: {len(y):,}  default rate {_pct(y.mean())}")
    print(f"Split (seed {art.seed}): train {len(tr):,} / valid {len(va):,} / test {len(te):,}")
    print(f"Features ({len(FEATURE_KEYS)}), with monotonic constraint:")
    for spec in FEATURES:
        print(f"  {spec.monotone:+d}  {spec.key}")
    print(f"Champion: XGBoost {model.CHAMPION_PARAMS}")
    print(
        f"  AUC train {evaluate.auc(y[tr], art.pd[tr]):.4f}"
        f"  valid {evaluate.auc(y[va], art.pd[va]):.4f}"
    )
    print("Benchmark: logistic regression")
    print(
        f"  AUC train {evaluate.auc(y[tr], art.pd_benchmark[tr]):.4f}"
        f"  valid {evaluate.auc(y[va], art.pd_benchmark[va]):.4f}"
    )
    a, b = art.platt
    print(
        f"Calibration check on validation: slope {a:.3f}, intercept {b:+.3f} "
        f"(tolerance {model.CALIBRATION_TOLERANCE})"
    )
    if art.calibration:
        print("  Platt scaling applied.")
    else:
        print("  Within tolerance, raw scores kept.")
    # XGBoost's own format, for anyone who wants to load the model elsewhere.
    saved = (Path(args.data_dir) if args.data_dir else default_data_dir()) / "champion_xgb.json"
    art.champion.save_model(saved)
    print(f"Saved the champion in XGBoost's native format to {saved}")
    return {}


def _print_evaluation(m: dict) -> None:
    c, b = m["champion"], m["benchmark"]
    print("Test set")
    print(f"  {'model':<22}{'AUC':>8}{'Gini':>8}{'KS':>8}")
    print(f"  {'XGBoost champion':<22}{c['auc']:>8.4f}{c['gini']:>8.4f}{c['ks']:>8.4f}")
    print(f"  {'Logistic benchmark':<22}{b['auc']:>8.4f}{b['gini']:>8.4f}{b['ks']:>8.4f}")
    print(f"  Champion AUC 95% bootstrap interval: {c['auc_ci_low']:.4f} to {c['auc_ci_high']:.4f}")
    d = m["champion_minus_benchmark_auc"]
    print(
        f"  Champion minus benchmark: {d['difference']:+.4f} "
        f"({d['ci_low']:+.4f} to {d['ci_high']:+.4f})"
    )
    print(
        f"  AUC train {m['auc_train']:.4f}, valid {m['auc_valid']:.4f}, "
        f"train minus test {m['train_test_gap']:+.4f}"
    )
    w = m["with_protected"]
    print(
        f"  With sex, age, marriage, education added: AUC {w['auc']:.4f} "
        f"(gain {w['auc_gain_over_champion']:+.4f}, "
        f"{w['gain_ci_low']:+.4f} to {w['gain_ci_high']:+.4f})"
    )
    u = m["unconstrained"]
    print(
        f"  Without monotonic constraints: AUC {u['auc']:.4f} "
        f"(constraints cost {u['auc_cost_of_constraints']:+.4f})"
    )
    print("\nDeciles (1 = lowest predicted risk)")
    table = pd.DataFrame(m["deciles"])[["decile", "n", "defaults", "default_rate", "avg_score"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if m["deciles_monotone"]:
        print("  Default rate rises in every decile.")
    else:
        print(f"  Default rate falls instead of rising at decile(s): {m['decile_inversions']}")
    cal = m["calibration"]
    print("\nCalibration")
    print(
        f"  Validation slope {cal['platt_slope_valid']:.3f}, "
        f"intercept {cal['platt_intercept_valid']:+.3f}; "
        f"{'Platt scaling applied' if cal['applied'] else 'no recalibration applied'}"
    )
    t = cal["test_as_used"]
    print(
        f"  Test: mean predicted {_pct(t['mean_predicted'], 2)}, "
        f"observed {_pct(t['observed_rate'], 2)}, "
        f"Brier {t['brier']:.4f}, expected calibration error {t['expected_calibration_error']:.4f}"
    )
    p = cal["test_if_platt_applied"]
    print(
        f"  Test if Platt scaling were applied: Brier {p['brier']:.4f}, "
        f"expected calibration error {p['expected_calibration_error']:.4f}"
    )
    r = m["reasons"]["agreement_with_pred_contribs"]
    print(f"\nReason codes against XGBoost pred_contribs, {r['n_compared']} declined test accounts")
    print(
        f"  Same top reason {_pct(r['top1_identical'])}; top reason within their top 4 "
        f"{_pct(r['top1_in_shap_top4'])}; mean overlap of the top 4 {_pct(r['mean_top4_overlap'])}"
    )


def cmd_evaluate(args) -> dict:
    metrics = evaluate_all(_build(args), args.n_boot)
    _print_evaluation(metrics)
    return metrics


def _print_cutoff(art: Artifacts, sensitivity: list[dict]) -> None:
    rec = art.recommended_cutoff
    print(
        f"Assumptions (illustrative): loss given default {_pct(art.lgd, 0)} of balance, "
        f"annual margin {_pct(art.margin_assumption, 0)} of balance on good accounts"
    )
    print(
        f"Break-even probability of default: {_pct(cutoff_mod.break_even_pd(art.lgd, art.margin_assumption))}"
    )
    cols = [
        "cutoff",
        "approval_rate",
        "default_rate_approved",
        "defaulters_declined",
        "defaulted_balance_declined",
        "goods_declined",
        "illustrative_profit_index",
    ]
    shown = art.cutoff_test[
        art.cutoff_test["cutoff"]
        .round(2)
        .isin(
            {
                0.05,
                0.08,
                0.10,
                0.12,
                0.14,
                0.16,
                0.18,
                0.20,
                0.25,
                0.30,
                0.40,
                0.50,
                0.60,
                round(rec, 2),
            }
        )
    ][cols]
    print("\nTest set, approve when predicted probability of default is below the cutoff")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    v, t = cutoff_mod.row_at(art.cutoff_valid, rec), cutoff_mod.row_at(art.cutoff_test, rec)
    print(f"\nRecommended cutoff: {rec:.2f} (maximises the profit view on the validation set)")
    print(
        f"  validation: approves {_pct(v['approval_rate'])}, declines {_pct(v['defaulters_declined'])} of defaulters"
    )
    print(
        f"  test:       approves {_pct(t['approval_rate'])}, declines {_pct(t['defaulters_declined'])} of defaulters "
        f"and {_pct(t['goods_declined'])} of good accounts; profit index {t['illustrative_profit_index']:.1f}"
    )
    print(f"  best cutoff on the test set itself: {cutoff_mod.best_cutoff(art.cutoff_test):.2f}")
    print("\nSensitivity: cutoff that maximises the profit view on validation, read on test")
    print(pd.DataFrame(sensitivity).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_cutoff(args) -> dict:
    art = _build(args)
    va, te = art.split.valid, art.split.test
    sens = cutoff_mod.sensitivity(
        art.y[va], art.pd[va], art.exposure[va], art.y[te], art.pd[te], art.exposure[te]
    )
    _print_cutoff(art, sens)
    return {}


def explain_row(art: Artifacts, row: int) -> dict:
    """Decision and reasons for the row-th account of the test set (0-based)."""
    test = art.split.test
    if not 0 <= row < len(test):
        raise SystemExit(f"--row must be between 0 and {len(test) - 1}")
    i = test[row]
    x = art.X.to_numpy()[i]
    return {
        "row": row,
        "probability": float(art.pd[i]),
        "cutoff": art.recommended_cutoff,
        "decision": "decline" if art.pd[i] >= art.recommended_cutoff else "approve",
        "observed_default": int(art.y[i]),
        "values": dict(zip(FEATURE_KEYS, map(float, x))),
        "reasons": reasons.explain(art.score, x, art.reference),
    }


def _print_explanation(result: dict, art: Artifacts) -> None:
    print(f"Test-set row {result['row']}")
    print(f"Predicted probability of default: {result['probability']:.4f}")
    print(f"Cutoff: {result['cutoff']:.2f}  ->  {result['decision'].upper()}")
    print(
        f"Observed outcome in the data: {'defaulted' if result['observed_default'] else 'did not default'}"
    )
    if result["decision"] == "decline":
        print("\nReasons for the decline, largest effect first:")
    else:
        print(
            "\nApproved, so no adverse-action reasons apply. Largest risk factors, for information:"
        )
    if not result["reasons"]:
        print("  none: no single feature raises the probability by 0.001 or more")
    for rank, r in enumerate(result["reasons"], start=1):
        print(f"  {rank}. {r['reason']}")
        print(
            f"     {r['feature']} = {r['value']:g}; typical approved account = {r['reference']:g}; "
            f"replacing it lowers the probability by {r['probability_drop']:.4f}"
        )
    print("\nAll inputs (applicant / typical approved account):")
    for key, ref in zip(FEATURE_KEYS, art.reference):
        print(f"  {key:<20} {result['values'][key]:>12g} / {float(ref):g}")


def cmd_explain(args) -> dict:
    art = _build(args)
    result = explain_row(art, args.row)
    _print_explanation(result, art)
    return {}


def _print_fairness(f: dict) -> None:
    print(f"Approval by group on the test set at cutoff {f['cutoff']:.2f}")
    for title, key, air in [
        ("Sex", "by_sex", "adverse_impact_ratio_sex"),
        ("Age band", "by_age_band", "adverse_impact_ratio_age"),
        ("Marital status", "by_marital_status", "adverse_impact_ratio_marital_status"),
    ]:
        print(f"\n{title}")
        print(pd.DataFrame(f[key]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        flag = "below" if f[air] < f["four_fifths_threshold"] else "above"
        print(
            f"  Adverse-impact ratio (lowest / highest approval rate): {f[air]:.3f}, {flag} the 0.80 screen"
        )
    if "age_gap_probe" in f:
        print("\nAge adverse-impact ratio after neutralising one feature at a time (top 3)")
        for row in f["age_gap_probe"][:3]:
            print(f"  {row['feature']:<20} {row['adverse_impact_ratio_age']:.3f}")
        limits = ", ".join(
            f"{k}: {v:,.0f}" for k, v in f["median_credit_limit_by_age_band"].items()
        )
        print(f"  Median credit limit by age band (NT$): {limits}")


def cmd_fairness(args) -> dict:
    art = _build(args)
    te = art.split.test
    raw, y, cutoff = art.raw.iloc[te], art.y[te], art.recommended_cutoff
    report = fairness.fairness_report(raw, y, art.pd[te], cutoff)
    report["age_gap_probe"] = fairness.age_gap_probe(
        art.score, art.X.to_numpy()[te], art.reference, FEATURE_KEYS, raw, y, cutoff
    )
    report["median_credit_limit_by_age_band"] = fairness.median_limit_by_age_band(raw)
    _print_fairness(report)
    return {}


def _print_psi(psi: dict) -> None:
    for title, key in [
        ("Train against test (no shift expected)", "train_vs_test"),
        ("Test against a deliberately shifted copy", "test_vs_shifted"),
    ]:
        print(f"\n{title}")
        print(pd.DataFrame(psi[key]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"\nThresholds: below {monitor.PSI_WATCH} stable, {monitor.PSI_WATCH} to {monitor.PSI_ACTION} investigate, "
        f"{monitor.PSI_ACTION} and above review"
    )


def cmd_monitor(args) -> dict:
    art = _build(args)
    tr, te = art.split.train, art.split.test
    X_test = art.X.iloc[te]
    X_shift = monitor.shifted_copy(X_test, art.seed)
    _print_psi(
        {
            "train_vs_test": monitor.psi_table(art.X.iloc[tr], X_test, art.pd[tr], art.pd[te]),
            "test_vs_shifted": monitor.psi_table(
                X_test, X_shift, art.pd[te], art.score(X_shift.to_numpy())
            ),
        }
    )
    return {}


def _export(art: Artifacts, metrics: dict, web_dir: Path) -> None:
    path = web_dir / "model.json"
    size = export.write_payload(export.build_payload(art, metrics), path)
    print(f"Wrote {path} ({size / 1000:.1f} kB)")


def cmd_export_web(args) -> dict:
    art = _build(args)
    _export(art, evaluate_all(art, args.n_boot), Path(args.web_dir))
    return {}


def cmd_all(args) -> dict:
    art = _build(args)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    cmd_train(args)
    print()
    metrics = evaluate_all(art, args.n_boot)
    _print_evaluation(metrics)
    print()
    _print_cutoff(art, metrics["cutoff"]["sensitivity"])
    print()
    _print_fairness(metrics["fairness"])
    _print_psi(metrics["psi"])
    print()

    te = art.split.test
    plots.plot_roc(
        art.y[te],
        art.pd[te],
        art.pd_benchmark[te],
        metrics["champion"]["auc"],
        metrics["benchmark"]["auc"],
        reports / "roc.png",
    )
    plots.plot_deciles(
        metrics["deciles"], metrics["split"]["default_rate_test"], reports / "deciles.png"
    )
    plots.plot_calibration(metrics["deciles"], reports / "calibration.png")
    plots.plot_cutoff(
        metrics["cutoff"]["grid_test"], art.recommended_cutoff, reports / "cutoff_tradeoff.png"
    )

    metrics_path = reports / "metrics.json"
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metrics_path} and 4 charts to {reports}")
    _export(art, metrics, Path(args.web_dir))
    return metrics


def cmd_validate(args) -> dict:
    """Review one submission, or all of them, with the rules baseline or the agent."""
    from .candidates import build_candidates
    from .validation import evidence_pack
    from .validator import chat_from_env, check_memo, rules_memo, run_agent, score_memo

    chat = chat_from_env() if args.agent else None
    if args.agent and chat is None:
        raise SystemExit("--agent needs VALIDATOR_LLM_API_KEY, VALIDATOR_LLM_BASE_URL and VALIDATOR_LLM_MODEL.")
    results = {}
    for c in build_candidates(args.seed, args.data_dir):
        if args.submission not in ("all", c.id):
            continue
        pack = evidence_pack(c)
        if chat is None:
            memo, calls = rules_memo(pack), len(pack)
            seen = {i["id"]: i["text"] for result in pack.values() for i in result["items"]}
        else:
            run = run_agent(pack, c.label, c.developer_summary, chat)
            memo, calls, seen = run.memo or {"decision": None, "summary": "", "findings": []}, len(run.calls), run.seen
        check, score = check_memo(memo, seen), score_memo(memo, c.planted)
        results[c.id] = {"memo": memo, "check": check.to_dict(), "score": score}
        print(f"\n{c.label}  ({'agent' if chat else 'rules baseline'}, {calls} test results read)")
        print(f"  Developer: {c.developer_summary}")
        print(f"  Decision: {memo['decision']}")
        for f in memo["findings"]:
            print(f"  [{f['severity']:6}] {f['title']}: {f['text']}  {' '.join('[' + e + ']' for e in f['evidence'])}")
        print(f"  Checker: {'PASS' if check.passed else 'FAIL ' + '; '.join(p['detail'] for p in check.problems)}")
        if args.reveal:
            caught = {None: "nothing was planted", True: "caught", False: "MISSED"}[score["caught"]]
            print(f"  Answer key: {c.planted.title}. {caught}. Decision defensible: {score['decision_ok']}.")
    return results


def cmd_score_runs(args) -> dict:
    """Re-score the published agent runs against the answer key and print the summary."""
    import statistics

    from .candidates import build_candidates
    from .validation import evidence_pack
    from .validator import check_memo, lookup, rules_memo, score_memo

    path = Path(args.reports_dir) / "agent_runs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data["runs"]
    rows, kinds = [], {}
    for c in build_candidates(args.seed, args.data_dir):
        pack = evidence_pack(c)
        mine = [r for r in runs if r["submission"] == c.id]
        for r in mine:
            seen = {}
            for entry in r["tests"]:  # "round:name" or "round:name:argument"
                parts = entry.split(":", 2)
                for item in lookup(pack, parts[1], parts[2] if len(parts) > 2 else None)["items"]:
                    seen[item["id"]] = item["text"]
            memo = r["memo"] or {}
            r["checker"] = check_memo(memo, seen).to_dict()
            r["score"] = score_memo(memo, c.planted)
            for problem in r["checker"]["problems"]:
                kinds[problem["kind"]] = kinds.get(problem["kind"], 0) + 1
        decisions: dict[str, int] = {}
        for r in mine:
            decision = (r["memo"] or {}).get("decision")
            decisions[decision] = decisions.get(decision, 0) + 1
        rules_decision = rules_memo(pack)["decision"]
        rows.append(
            {
                "id": c.id,
                "planted": c.planted.key,
                "title": c.planted.title,
                "runs": len(mine),
                "caught": None if c.planted.key == "none" else sum(1 for r in mine if r["score"]["caught"]),
                "decision_ok": sum(1 for r in mine if r["score"]["decision_ok"]),
                "checker_pass": sum(1 for r in mine if r["checker"]["passed"]),
                "false_alarms": sum(r["score"]["false_alarms"] for r in mine),
                "mean_calls": round(statistics.mean(r["test_runs"] for r in mine), 1),
                "decisions": decisions,
                "rules_decision": rules_decision,
                "agrees_with_rules": decisions.get(rules_decision, 0),
            }
        )

    def memos_with(kind: str) -> int:
        return sum(1 for r in runs if any(p["kind"] == kind for p in r["checker"]["problems"]))

    summary = {
        "run_on": data["about"].get("run_on_text", "22 November 2026"),
        "trials": max(r["trial"] for r in runs),
        "rows": rows,
        "totals": {
            "runs": len(runs),
            "defect_runs": sum(row["runs"] for row in rows if row["planted"] != "none"),
            "caught": sum(row["caught"] or 0 for row in rows),
            "decision_ok": sum(row["decision_ok"] for row in rows),
            "checker_pass": sum(row["checker_pass"] for row in rows),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in runs), 1),
            "mean_tokens": round(statistics.mean(r["input_tokens"] + r["output_tokens"] for r in runs)),
            "input_tokens": sum(r["input_tokens"] for r in runs),
            "output_tokens": sum(r["output_tokens"] for r in runs),
            "mean_rounds": round(statistics.mean(r["rounds"] for r in runs), 1),
            "agrees_with_rules": sum(row["agrees_with_rules"] for row in rows),
        },
        "problems": {
            "by_kind": kinds,
            "memos_with_citation_slip": memos_with("number_not_cited"),
            "memos_with_unsupported_number": memos_with("unsupported_number"),
            "memos_with_inconsistent_decision": memos_with("inconsistent"),
        },
    }
    out = Path(args.reports_dir) / "agent_eval.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    t = summary["totals"]
    print(f"{t['runs']} agent runs, {summary['trials']} per submission")
    print(f"  planted defect caught:        {t['caught']} of {t['defect_runs']}")
    print(f"  defensible decision:          {t['decision_ok']} of {t['runs']}")
    print(f"  same decision as fixed rules: {t['agrees_with_rules']} of {t['runs']}")
    print(f"  memo passed the checker:      {t['checker_pass']} of {t['runs']}")
    print(f"  checker problems by kind:     {kinds}")
    print(f"  mean run: {t['mean_seconds']} s, {t['mean_tokens']:,} tokens, {t['mean_rounds']} rounds")
    for row in rows:
        caught = "no false alarm" if row["caught"] is None and not row["false_alarms"] else f"{row['false_alarms']} false alarms" if row["caught"] is None else f"caught {row['caught']}/{row['runs']}"
        print(f"  {row['id']}  {row['title']:32} {caught:16} decisions {row['decisions']}  rules: {row['rules_decision']}")
    print(f"Wrote {out}")
    return summary


def cmd_export_validation(args) -> dict:
    from .export import build_validation_payload, write_validation_payload

    path = Path(args.web_dir) / "validation.json"
    size = write_validation_payload(build_validation_payload(args.seed), path)
    print(f"Wrote {path} ({size / 1000:.1f} kB)")
    return {"path": str(path), "bytes": size}


COMMANDS = {
    "train": (cmd_train, "fit the champion and benchmark, print the fit summary"),
    "evaluate": (cmd_evaluate, "test-set metrics, deciles, calibration, comparison models"),
    "cutoff": (cmd_cutoff, "cutoff table, recommended cutoff, sensitivity"),
    "explain": (cmd_explain, "decision and reason codes for one test-set row"),
    "fairness": (cmd_fairness, "approval rates by sex, age band and marital status"),
    "monitor": (cmd_monitor, "PSI against train and against a shifted copy"),
    "export-web": (cmd_export_web, "write web_export/model.json"),
    "validate": (cmd_validate, "review a submission with the rules baseline or the validation agent"),
    "export-validation": (cmd_export_validation, "write web_export/validation.json for the agent demo"),
    "score-runs": (cmd_score_runs, "re-score the published agent runs in reports/agent_runs.json"),
    "all": (cmd_all, "run everything and write reports/ and web_export/"),
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m creditadj",
        description="Credit risk adjudication model on the UCI credit card default data.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=SEED, help="split and model seed")
    common.add_argument(
        "--lgd",
        type=float,
        default=cutoff_mod.DEFAULT_LGD,
        help="assumed loss given default, share of balance",
    )
    common.add_argument(
        "--margin",
        type=float,
        default=cutoff_mod.DEFAULT_MARGIN,
        help="assumed annual margin on good balances",
    )
    common.add_argument(
        "--data-dir", default=None, help="where the dataset is cached (default: data/)"
    )
    common.add_argument(
        "--n-boot", type=int, default=1000, help="bootstrap resamples for intervals"
    )
    common.add_argument("--reports-dir", default=str(ROOT / "reports"))
    common.add_argument("--web-dir", default=str(ROOT / "web_export"))

    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        p = sub.add_parser(name, parents=[common], help=description)
        if name == "explain":
            p.add_argument("--row", type=int, required=True, help="0-based row of the test set")
        if name == "validate":
            p.add_argument("--submission", default="all", help="A to H, or all")
            p.add_argument("--agent", action="store_true", help="use the language-model agent instead of the rules baseline")
            p.add_argument("--reveal", action="store_true", help="show the answer key after each review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        COMMANDS[args.command][0](args)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader went away (for example `| head`). Point stdout at
        # /dev/null so Python does not complain again while shutting down.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return 0
