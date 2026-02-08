from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Core engine (standalone)
# ----------------------------

@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    category: str
    passed: bool
    hard: bool
    penalty: float
    weight: float
    message: str


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    total_risk: float
    disqualified: bool
    results: List[RuleResult]
    fingerprint: str


def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _get_field(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _penalty_bool(passed: bool) -> float:
    return 0.0 if passed else 1.0


def _penalty_out_of_range(value: float, lo: float, hi: float) -> Tuple[bool, float, str]:
    if lo <= value <= hi:
        return True, 0.0, f"value={value} within [{lo},{hi}]"
    rng = max(hi - lo, 1e-12)
    if value < lo:
        dist = (lo - value) / rng
        return False, min(1.0 + dist, 5.0), f"value={value} below lo={lo} (dist={dist:.3f} ranges)"
    dist = (value - hi) / rng
    return False, min(1.0 + dist, 5.0), f"value={value} above hi={hi} (dist={dist:.3f} ranges)"


def _safe_eval(expr: str, candidate: Dict[str, Any], context: Dict[str, Any]) -> Any:
    safe_names = {
        "candidate": candidate,
        "context": context,
        "math": math,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "len": len,
        "all": all,
        "any": any,
    }
    return eval(expr, {"__builtins__": {}}, safe_names)


def evaluate_candidate(
    candidate: Dict[str, Any],
    rules: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> CandidateResult:
    context = context or {}
    results: List[RuleResult] = []
    disqualified = False

    for r in rules:
        rule_id = str(r["id"])
        category = str(r.get("category", "general"))
        weight = float(r.get("weight", 1.0))
        hard = bool(r.get("hard", False))
        rtype = str(r["type"])
        field = str(r.get("field", ""))
        params = dict(r.get("params", {}))
        label = str(r.get("message", rule_id))

        passed = True
        penalty = 0.0
        detail = ""

        try:
            if rtype == "required_field":
                val = _get_field(candidate, field)
                passed = val is not None
                penalty = _penalty_bool(passed)
                detail = f"field '{field}' present={passed}"

            elif rtype == "range":
                val = _get_field(candidate, field)
                if val is None:
                    passed = False
                    penalty = 1.0
                    detail = f"field '{field}' missing"
                else:
                    lo = float(params["min"])
                    hi = float(params["max"])
                    passed, penalty, detail = _penalty_out_of_range(float(val), lo, hi)

            elif rtype == "equals":
                val = _get_field(candidate, field)
                expected = params.get("value")
                passed = val == expected
                penalty = _penalty_bool(passed)
                detail = f"value={val!r} expected={expected!r}"

            elif rtype == "custom":
                expr = str(params["expr"])
                out = _safe_eval(expr, candidate, context)
                passed = bool(out)
                penalty = _penalty_bool(passed)
                detail = f"expr -> {out!r}"

            else:
                passed = False
                penalty = 1.0
                detail = f"unknown rule type '{rtype}'"

        except Exception as e:
            passed = False
            penalty = 2.0
            detail = f"rule error: {type(e).__name__}: {e}"

        if hard and not passed:
            disqualified = True

        results.append(
            RuleResult(
                rule_id=rule_id,
                category=category,
                passed=passed,
                hard=hard,
                penalty=float(penalty),
                weight=float(weight),
                message=f"{label}. {detail}",
            )
        )

    total_risk = sum(rr.penalty * rr.weight for rr in results)
    candidate_id = str(candidate.get("candidate_id") or _stable_hash(candidate))
    fingerprint = _stable_hash({"candidate": candidate, "rules": rules, "context": context})

    return CandidateResult(
        candidate_id=candidate_id,
        total_risk=float(total_risk),
        disqualified=disqualified,
        results=results,
        fingerprint=fingerprint,
    )


def select_least_risky(all_results: List[CandidateResult]) -> CandidateResult:
    eligible = [r for r in all_results if not r.disqualified]
    if eligible:
        return min(eligible, key=lambda r: r.total_risk)
    return min(all_results, key=lambda r: r.total_risk)


def build_audit(selected: CandidateResult, all_results: List[CandidateResult]) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate_id": selected.candidate_id,
        "selected_total_risk": selected.total_risk,
        "selected_disqualified": selected.disqualified,
        "selection_fingerprint": selected.fingerprint,
        "candidates": [
            {
                "candidate_id": r.candidate_id,
                "total_risk": r.total_risk,
                "disqualified": r.disqualified,
                "fingerprint": r.fingerprint,
                "rule_results": [
                    {
                        "rule_id": rr.rule_id,
                        "category": rr.category,
                        "passed": rr.passed,
                        "hard": rr.hard,
                        "penalty": rr.penalty,
                        "weight": rr.weight,
                        "contribution": rr.penalty * rr.weight,
                        "message": rr.message,
                    }
                    for rr in r.results
                ],
            }
            for r in all_results
        ],
    }


# ----------------------------
# Streamlit GUI (standalone)
# ----------------------------

st.set_page_config(page_title="Least-Risk Candidate Selector", layout="wide")
st.title("Least-Risk Candidate Selector")
st.caption("Paste multiple candidate outputs + a ruleset. The app verifies, scores risk, and selects the least-risk candidate.")

left, right = st.columns([1, 2], gap="large")

DEFAULT_RULES = [
    {
        "id": "req_candidate_id",
        "category": "schema",
        "weight": 3.0,
        "hard": True,
        "type": "required_field",
        "field": "candidate_id",
        "message": "Candidate must have candidate_id",
    },
    {
        "id": "prob_exploit_bounds",
        "category": "bounds",
        "weight": 4.0,
        "hard": True,
        "type": "range",
        "field": "exploit_probability",
        "params": {"min": 0.0, "max": 1.0},
        "message": "Exploit probability must be within [0,1]",
    },
    {
        "id": "prob_exposure_bounds",
        "category": "bounds",
        "weight": 4.0,
        "hard": True,
        "type": "range",
        "field": "enterprise_exposure_probability",
        "params": {"min": 0.0, "max": 1.0},
        "message": "Exposure probability must be within [0,1]",
    },
    {
        "id": "recommended_action_required",
        "category": "schema",
        "weight": 2.0,
        "hard": True,
        "type": "required_field",
        "field": "recommended_action",
        "message": "Candidate must have recommended_action",
    },
    {
        "id": "prefer_lower_joint_risk",
        "category": "preference",
        "weight": 1.0,
        "hard": False,
        "type": "custom",
        "params": {
            "expr": "(candidate.get('exploit_probability', 1.0) * candidate.get('enterprise_exposure_probability', 1.0)) <= 0.20"
        },
        "message": "Prefer lower joint probability (exploit × exposure ≤ 0.20)",
    },
]

DEFAULT_CANDIDATES = [
    {
        "candidate_id": "A",
        "exploit_probability": 0.25,
        "enterprise_exposure_probability": 0.125,
        "recommended_action": "PatchNow",
        "model_meta": {"model": "alpha", "temperature": 0.2},
    },
    {
        "candidate_id": "B",
        "exploit_probability": 0.65,
        "enterprise_exposure_probability": 0.60,
        "recommended_action": "Investigate",
        "model_meta": {"model": "beta", "temperature": 0.7},
    },
    {
        "candidate_id": "C",
        "exploit_probability": 1.20,
        "enterprise_exposure_probability": 0.10,
        "recommended_action": "Defer",
        "model_meta": {"model": "gamma", "temperature": 0.9},
    },
]

with left:
    st.subheader("Inputs")

    rules_text = st.text_area(
        "Rules (JSON list)",
        value=json.dumps(DEFAULT_RULES, indent=2),
        height=230,
    )

    candidates_text = st.text_area(
        "Candidates (JSON list)",
        value=json.dumps(DEFAULT_CANDIDATES, indent=2),
        height=260,
    )

    context_text = st.text_area(
        "Optional context (JSON object)",
        value="{}",
        height=100,
    )

    st.divider()
    st.write("Scoring knobs")
    disqualify_mode = st.checkbox("Disqualify candidates that fail any hard rule", value=True)
    hard_fail_penalty = st.number_input("Extra penalty for hard failures (if not disqualifying)", min_value=0.0, value=50.0, step=1.0)

    run = st.button("Run verification + selection", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if not run:
        st.info("Edit inputs, then click **Run verification + selection**.")
        st.stop()

    try:
        rules = json.loads(rules_text)
        candidates = json.loads(candidates_text)
        context = json.loads(context_text)
        if not isinstance(rules, list):
            raise ValueError("Rules must be a JSON list.")
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a JSON list.")
        if not isinstance(context, dict):
            raise ValueError("Context must be a JSON object.")
    except Exception as e:
        st.error(f"Parse error: {e}")
        st.stop()

    evaluated: List[CandidateResult] = []
    for c in candidates:
        r = evaluate_candidate(c, rules, context=context)
        if (not disqualify_mode) and r.disqualified:
            # keep candidate eligible but slam it with a big penalty to make it very unlikely to win
            boosted = CandidateResult(
                candidate_id=r.candidate_id,
                total_risk=r.total_risk + hard_fail_penalty,
                disqualified=False,
                results=r.results,
                fingerprint=r.fingerprint,
            )
            evaluated.append(boosted)
        else:
            evaluated.append(r)

    selected = select_least_risky(evaluated)

    # Summary dataframe
    summary_rows = []
    for r in evaluated:
        fails = sum(1 for rr in r.results if not rr.passed)
        hard_fails = sum(1 for rr in r.results if (rr.hard and not rr.passed))
        summary_rows.append(
            {
                "candidate_id": r.candidate_id,
                "total_risk": r.total_risk,
                "failed_rules": fails,
                "hard_failed_rules": hard_fails,
                "disqualified": (disqualify_mode and any(rr.hard and (not rr.passed) for rr in r.results)),
                "selected": r.candidate_id == selected.candidate_id,
            }
        )
    df_summary = pd.DataFrame(summary_rows).sort_values(["selected", "total_risk"], ascending=[False, True])

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Selected", selected.candidate_id)
    with c2:
        st.metric("Selected risk", f"{selected.total_risk:.3f}")
    with c3:
        st.caption("Lower risk wins. Hard-rule failures can disqualify or be heavily penalized based on the setting.")

    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # Build heatmap table
    heat_rows = []
    for r in evaluated:
        for rr in r.results:
            heat_rows.append(
                {
                    "candidate_id": r.candidate_id,
                    "rule_id": rr.rule_id,
                    "category": rr.category,
                    "passed": rr.passed,
                    "hard": rr.hard,
                    "penalty": rr.penalty,
                    "weight": rr.weight,
                    "contribution": rr.penalty * rr.weight,
                }
            )
    df_heat = pd.DataFrame(heat_rows)

    st.markdown("### Risk heatmap (contribution = penalty × weight)")
    fig_heat = px.density_heatmap(
        df_heat,
        x="rule_id",
        y="candidate_id",
        z="contribution",
        hover_data=["category", "passed", "hard", "penalty", "weight"],
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Risk composition by category (stacked)")
    df_cat = df_heat.groupby(["candidate_id", "category"], as_index=False)["contribution"].sum()
    fig_cat = px.bar(df_cat, x="candidate_id", y="contribution", color="category", barmode="stack")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("### Drill-down")
    selected_id = st.selectbox(
        "Select candidate to inspect",
        options=list(df_summary["candidate_id"]),
        index=list(df_summary["candidate_id"]).index(selected.candidate_id),
    )

    sel_obj = next(r for r in evaluated if r.candidate_id == selected_id)
    df_sel = pd.DataFrame(
        [
            {
                "rule_id": rr.rule_id,
                "category": rr.category,
                "passed": rr.passed,
                "hard": rr.hard,
                "penalty": rr.penalty,
                "weight": rr.weight,
                "contribution": rr.penalty * rr.weight,
                "message": rr.message,
            }
            for rr in sel_obj.results
        ]
    ).sort_values(["hard", "passed", "contribution"], ascending=[False, True, False])

    st.dataframe(df_sel, use_container_width=True, hide_index=True)

    st.markdown("### Audit export")
    audit = build_audit(selected, evaluated)
    st.download_button(
        "Download audit JSON",
        data=json.dumps(audit, indent=2),
        file_name="least_risk_selection_audit.json",
        mime="application/json",
        use_container_width=True,
    )
