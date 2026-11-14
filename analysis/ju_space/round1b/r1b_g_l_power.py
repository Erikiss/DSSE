"""Part L's predeclared under-power guard for equivalence conclusions.

Planning values below come from the L plan's MDE paragraphs. Where only an MDE
is stated, its explicitly defined factor 2.49 converts it to planning SE. No
planning SE is inferred from a margin or an observed effect. Missing planning
or realized SE prevents an equivalence conclusion; directional tests survive.
"""
from __future__ import annotations

import copy
import math


def assessment(stat, planning_se, source):
    value = stat.get("se") if isinstance(stat, dict) else None
    value = float(value) if (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and value >= 0) else None
    result = {"planning_se": planning_se, "realised_se": value, "source": source, "threshold_multiplier": 1.5}
    if planning_se is None:
        result.update(status="planning_SE_unavailable", passed=None, replacement="not decidable (planning SE unavailable)")
    elif value is None:
        result.update(status="realised_SE_unavailable", passed=None, replacement="not decidable (realised SE unavailable)")
    elif value > 1.5 * planning_se:
        result.update(status="underpowered", passed=False, replacement="not decidable (underpowered)")
    else:
        result.update(status="passed", passed=True, replacement=None)
    return result


def guard(verdicts, statistics):
    """Return guarded verdicts and auditable checks, changing equivalence only."""
    output, checks = copy.deepcopy(verdicts), {}

    def apply(branch, field, expected, requirements):
        verdict = output.get(branch, {})
        if not isinstance(verdict, dict) or verdict.get(field) not in expected:
            return
        results = {}
        for name, planning_se, source in requirements:
            data = statistics.get(branch, {})
            for key in name.split("."):
                data = data.get(key, {}) if isinstance(data, dict) else {}
            results[name] = assessment(data, planning_se, source)
        checks[f"{branch}.{field}"] = results
        failed = [record for record in results.values() if record["passed"] is not True]
        if failed:
            verdict[field] = next((r["replacement"] for r in failed if r["status"] == "underpowered"), failed[0]["replacement"])

    apply("L1_erasure", "privilege", {"H not causally privileged"}, [
        ("diff_ci_wrong", .05 / 2.49, "L1 erasure MDE .05; L.2 MDE=2.49 planning SE"),
        ("diff_fab_nonref", .05 / 2.49, "L1 erasure MDE .05; L.2 MDE=2.49 planning SE")])
    apply("L1_erasure", "curve", {"robust up to f = 1/16"}, [
        ("curve_total_change", None, "L1 states a pointwise curve margin but no curve-specific planning SE")])
    apply("L1_erasure", "curve", {"threshold-like within f <= 1/16"}, [
        ("curve_prethreshold_change", None,
         "L1 threshold requires pre-threshold equivalence within .04 but gives no curve-specific planning SE")])
    apply("L2a", "own", {"no deficit"}, [("lnR_top", .05, "L2a planning SE of ln R is .05")])
    apply("L2b", "verdict", {"length effect on H is information-limited"}, [("bie", .02, "L2b slope planning SE .02")])
    apply("L2b", "verdict", {"no length trend of H"}, [("b0", .02, "L2b slope planning SE .02")])
    apply("L3b", "specificity", {"carry like damage-matched controls"}, [
        ("dK", .136, "L3b planning sampling SD of Delta K .136")])
    apply("L3d", "ebm", {"no energy signature at entry"}, [
        ("de", .035, "L3d Delta e planning SE .035"), ("dtt", .10, "L3d Delta TT planning SE .10")])
    apply("L3d", "irrev", {"irreversibility not entry-specific"}, [
        ("irr_a.post_minus_match", None, "L3d specifies a realized resolution band, not a planning SE")])
    apply("L3d", "irrev", {"reversible within resolution"}, [
        ("irr_a.post_minus_rev", None, "L3d specifies a realized resolution band, not a planning SE")])
    apply("L4", "pair", {"no detectable inter-agent coupling"}, [("J_ex", .04, "L4 J_ex planning SE .04")])
    apply("L4", "collaboration", {"influence explained by lexical priming (no social coupling)"}, [
        ("J_ex_minus_doc", .055, "L4 collaboration contrast planning SE .055")])
    apply("L4", "h_line", {"H gain acts like damage"}, [("DD", .058, "L4 DD planning SE .058")])
    apply("L4", "h_line", {"no effect of H gain on coupling"}, [
        ("DD", .058, "L4 DD planning SE .058"),
        ("hup_minus_half", None, "L4 gives DD planning SE but no separate Hup-minus-half planning SE")])
    apply("L4", "catfish", {"no catfish effect"}, [("catfish", .06, "L4 log K ratio planning SE .06")])
    apply("L4", "macro", {"no pair macro beyond uncoupled agents"}, [
        ("dpsi", None, "L4 specifies the macro margin but no planning SE")])
    apply("L5a", "verdict", {"no growth of false readouts with chain length"}, [
        ("G_all", None, "L5a gives G_F(known) planning precision, not combined G_all planning SE")])
    apply("L5a", "verdict", {"break-off owned by the model"}, [
        ("GF_known", .2 / math.sqrt(300), "L5a planned per-item SD .2 over300 independent known items")])
    apply("L5b", "verdict", {"H-pattern owned by the model"}, [
        ("rho_w", .035 / 2.49, "L5b rho_w MDE .035; L.2 MDE=2.49 planning SE")])
    apply("L5b", "verdict", {"H-pattern owned by the lens"}, [("rho_b", .049, "L5b rho_b planning SE .049")])
    apply("L5b", "verdict", {"no H-pattern"}, [
        ("rho_b", .049, "L5b rho_b planning SE .049"),
        ("rho_w", .035 / 2.49, "L5b rho_w MDE .035; L.2 MDE=2.49 planning SE")])
    apply("L5c", "verdict", {"no H signal before own errors"}, [("beta_H", .05, "L5c planning SE .05")])

    # These are equivalence qualifiers, not replacements for the primary result.
    for branch, label, names in (
        ("L1", "neuron basis not privileged", ("rot_minus_U",)),
        ("L2b", "generic (not H-specific) length trend", ("db0",)),
    ):
        verdict = output.get(branch, {})
        if isinstance(verdict, dict) and label in verdict.get("labels", []):
            record = assessment(None, None, f"{branch} does not specify the planning SE of this contrast")
            checks[f"{branch}.label:{label}"] = {name: record for name in names}
            verdict["labels"] = [v for v in verdict["labels"] if v != label] + [f"{label}: equivalence not decidable (planning SE unavailable)"]
    return output, checks
