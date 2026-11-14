"""Per-target G confirmatory families from independently collected K/L evidence.

The frozen main plan section 2.4 controls CF-K-attr, CF-K-link, CF-L and
CF-HERR. Part-level strings stay unchanged; this final CPU stage adds their
round-level labels. Other models are never pooled into these families.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import r1b_core as K
from r1b_k_base import holm_family


SOURCES = {
    "detection": ("k_analyze", "k3_verdicts.json"),
    "chain": ("k_chain_analyze", "chain_analysis.json"),
    "persistence": ("k_persistence", "persistence_verdicts.json"),
    "persistence_status": ("k_persistence", "persistence_status.json"),
    "extra": ("k_extra", "extra_analysis.json"),
    "long": ("k_long", "long_verdicts.json"),
    "owner": ("l_owner", "l5_verdicts.json"),
    "l_analysis": ("l_analyze", "l_analysis.json"),
    "l3a": ("l_dynamics", "l3a.json"),
    "l3b": ("l_dynamics", "l3b.json"),
    "l2b": ("l_dynamics", "l2b.json"),
}

ATTRACTOR_MEMBERS = (
    "K6e TOK carry", "K6e TOK+KV carry", "K6e bistability",
    "K8b ACH carry", "K8b TQA carry", "K8b ACH bistability",
)
LINK_MEMBERS = (
    "K6b(i)", "K6b(ii)", "K6c(i)", "K6c(ii)",
    "K6e-c push", "K6e-c clamp", "K6h",
)


def _p(value):
    """Missing/non-finite/coded p-values cannot supply positive evidence."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
        return float(value)
    return 1.0


def _string(value):
    return value if isinstance(value, str) else value.get("verdict", "") if isinstance(value, dict) else ""


def _passed(gate):
    return gate is True or isinstance(gate, dict) and gate.get("pass") is True


def _member(string, p, *, reached=False, gates=(), available=True, mock=False):
    failed = [name for name, passed in gates if not passed]
    if mock:
        failed.append("MOCK is not scientific evidence")
    if not available:
        failed.append("measurement unavailable")
    eligible = bool(reached and not failed)
    return {"part_verdict": string or "not testable (measurement unavailable)",
            "raw_p": _p(p), "p": _p(p) if eligible else 1.0,
            "part_rule_reached": bool(reached), "eligible": eligible,
            "closed_gates": failed}


def _family(name, members):
    adjusted = holm_family({key: row["p"] for key, row in members.items()})
    result = {}
    for key, row in members.items():
        h = adjusted[key]
        confirmed = bool(row["eligible"] and h["reject"])
        result[key] = {**row, **h, "confirmed": confirmed,
                       "label": f"confirmatory ({name})" if confirmed else
                                "not testable (gate closed or measurement unavailable)" if row["closed_gates"] else
                                "exploratory (part verdict not reached)" if not row["part_rule_reached"] else
                                "exploratory (Holm not met)"}
    return {"alpha": .05, "method": "Holm", "size": len(result), "members": result,
            "any_confirmatory": any(row["confirmed"] for row in result.values())}


def summarize(sources, *, mock, model):
    """Pure family assembly; no model calls, fitting or new hypothesis tests."""
    det, chain, persistence, extra, long, owner, la = (
        sources.get(key) or {} for key in ("detection", "chain", "persistence", "extra", "long", "owner", "l_analysis"))
    k3 = det.get("K3a") or {}
    arithmetic = ("transfers to arithmetic errors" in (k3.get("arith") or {}).get("string", "") or
                  (k3.get("har") or {}).get("string") == "H_ar replicated (arithmetic in-domain)")
    sg8 = det.get("SG8") is True
    chain_gates = chain.get("gates") or {}
    # G-K7 is label-only (K.6); it does not close a confirmatory member.
    lens_gates = [(name, _passed(chain_gates.get(name))) for name in ("G_K_J", "G_K4")]
    stats = chain.get("statistics") or {}
    causal = stats.get("K6c_provisional") or {}
    lens_persistence = persistence.get("K6e") or {}
    persistence_status = sources.get("persistence_status") or {}
    persistence_missing = persistence_status.get("missing") or {}
    qa_branch = (extra.get("branches") or {}).get("K6h") or {}
    qa = qa_branch.get("statistics") or {}
    q12 = qa.get("Q12") or {}
    q12_positive = qa.get("G_K23", "").startswith("lens reads") and _p(q12.get("p_iut")) <= .05
    q12_string = "H-activity predicts lens confabulation beyond chain length (QA)" if q12_positive else q12.get("string", "undecided")
    causal_e = lens_persistence.get("causal") or {}
    rows = {
        "K6b(i)": stats.get("K6b_i_provisional") or {},
        "K6b(ii)": stats.get("K6b_ii_provisional") or {},
        "K6c(i)": causal.get("(i)") or {},
        "K6c(ii)": causal.get("(ii)") or {},
        "K6e-c push": causal_e.get("PUSH_H") or {},
        "K6e-c clamp": causal_e.get("CLAMP_H") or {},
        "K6h": q12,
    }
    link, rank = {}, {}
    for name in LINK_MEMBERS:
        row = rows[name]
        if name == "K6h":
            string, reached = q12_string, q12_positive
            # QA has its own positive control; an arithmetic K4 failure must
            # not suppress its independently validated QA lens branch.
            gates = [("G_K_J", _passed(qa_branch.get("G_K_J"))),
                     ("SG8 (QA H-set)", sg8), ("G_K23", qa.get("G_K23", "").startswith("lens reads"))]
        elif name.startswith("K6e-c"):
            condition = "PUSH_H" if name.endswith("push") else "CLAMP_H"
            string = (persistence.get("strings") or {}).get("K6e " + condition, "")
            reached = string.startswith("H push") or string.startswith("H clamp lowers")
            gates = lens_gates + [("G_K3 (arithmetic H-set)", arithmetic),
                                  ("K6e sampling/calibration", bool(persistence_status) and not persistence_missing.get("K6e"))]
        else:
            string = row.get("string", "")
            reached = string.startswith("H-activity") if name.startswith("K6b") else "reduces lens confabulation specifically" in string
            gates = lens_gates + [("G_K3 (arithmetic H-set)", arithmetic)]
        # Main §2.4 selects the PI specificity form. Q11 already uses PI.
        raw_pi = row.get("p_iut") if name.startswith("K6e-c") else row.get("p_iut_pi")
        link[name] = _member(string, raw_pi, reached=reached, gates=gates, available=bool(row), mock=mock)
        rank[name] = _member(string, row.get("p_iut"), reached=reached, gates=gates, available=bool(row), mock=mock)
    attr = {}
    for name in ATTRACTOR_MEMBERS:
        string = (persistence.get("strings") or {}).get(name, "")
        reached = (string.startswith("H-neuron attractor") or string.startswith("self-sustaining H") or
                   string == "within-item bistability (sessions)")
        branch = "K6e" if name.startswith("K6e") else "K8b"
        gates = [("G_K3 (arithmetic H-set)", arithmetic)] if "TQA" not in name else [("SG8 (QA H-set)", sg8)]
        gates += [(f"{branch} sampling/calibration", bool(persistence_status) and not persistence_missing.get(branch))]
        if branch == "K6e":
            gates += lens_gates
        q9 = (persistence.get("Q9") or {}).get(name)
        attr[name] = _member(string, (q9 or {}).get("p"), reached=reached, gates=gates, available=q9 is not None, mock=mock)
    long_string = long.get("string", "")
    lv = (owner.get("verdicts") or {}).get("L5c") or {}
    own_string = _string(lv)
    herr = {
        "K8c": _member(long_string, (long.get("rule") or {}).get("p_iut"),
                        reached=long_string.startswith("H-activity predicts the next error"),
                        gates=[("G_K3 (arithmetic H-set)", arithmetic),
                               ("G_K13 (long-form format)", long.get("G_K13") is None and _p(long.get("parse_fail")) <= .20)],
                        available=bool(long), mock=mock),
        "L5c": _member(own_string, owner.get("CF_HERR_member"), reached=own_string.startswith("H predicts"),
                        available=bool(lv), mock=mock),
    }
    l_verdicts = la.get("verdicts") or {}
    l_attr = l_verdicts.get("attractor") or {"verdict": "not testable (measurement unavailable)"}
    l2b = l_verdicts.get("L2b") or {"verdict": "not testable (measurement unavailable)"}
    # L3b supplies one alternative attractor route; the L3a bistability route
    # can reach the same fixed member independently. Missing route p=1 below.
    l_sources_present = bool(sources.get("l_analysis") and sources.get("l3a"))
    l_error = None
    try:
        from r1b_l_rules import cf_l_p
        # Sanitize missing/coded statistics to p=1 separately, so unavailable
        # attractor evidence cannot suppress an independently measured L2b.
        a, b, c = (sources.get(k) or {} for k in ("l3a", "l3b", "l2b"))
        safe_a = {k: _p(a.get(k)) for k in ("bi_sur_p", "bi_rank_p", "gamma_sur_p", "dgamma_rank_p", "hd_rank_p")}
        safe_a.update({k: {"p_gt0": _p((a.get(k) or {}).get("p_gt0"))}
                       for k in ("bi_minus_sel", "dgamma_minus_sel", "hd_minus_sel")})
        safe_b = {"K_pool": {"p_gt1": _p((b.get("K_pool") or {}).get("p_gt1"))},
                  "dK": {"p_gt0": _p((b.get("dK") or {}).get("p_gt0"))}}
        safe_c = {k: {"p_gt0": _p((c.get(k) or {}).get("p_gt0"))} for k in ("bie", "dbie")}
        lp = cf_l_p(safe_a, safe_b, safe_c,
                    l_verdicts.get("L3a") or {}, l_verdicts.get("L3b") or {}, l2b, l_attr)["p"]
    except (KeyError, TypeError, ValueError) as error:
        lp, l_error = {"attractor": 1., "l2b": 1.}, str(error)
    l_members = {
        "L3 attractor": _member(_string(l_attr), lp["attractor"],
                                reached=_string(l_attr) == "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)",
                                available=l_sources_present and l_error is None, mock=mock),
        "L2b": _member(_string(l2b), lp["l2b"], reached=_string(l2b) == "H rises beyond information loss",
                        available=bool(sources.get("l2b") and sources.get("l_analysis")) and l_error is None, mock=mock),
    }
    families = {"CF-K-attr": _family("CF-K-attr", attr), "CF-K-link": _family("CF-K-link", link),
                "CF-L": _family("CF-L", l_members), "CF-HERR": _family("CF-HERR", herr)}
    rank_family = _family("CF-K-link rank sensitivity", rank)
    rank_family.update(confirmatory=False, interpretation="descriptive only; main plan selects PI specificity")
    for value in rank_family["members"].values():
        value["confirmed"] = False
        value["label"] = "descriptive rank-form sensitivity"
    rank_family["any_confirmatory"] = False
    missing = [name for name in SOURCES if not sources.get(name)]
    return {"status": "partial" if missing else "done", "model": model, "mock": bool(mock),
            "families": families, "rank_sensitivity": rank_family,
            "scientific_evidence": not mock, "missing_sources": missing,
            "coverage": {name: {"status": "measured", "members": family["size"]} for name, family in families.items()},
            "interpretation": "MOCK: no confirmatory conclusions" if mock else
                              "Part strings are preserved; only reached, eligible members surviving Holm are confirmatory",
            "CF_L_input_issue": l_error,
            "qualifiers": {"G_K7": {"gate": chain_gates.get("G_K7"), "effect": "label only (K.6)"}},
            "scope": "separate families within this Gemma target; cross-model comparison remains descriptive",
            "exclusions": ["CF-R+/CF-R- are outside G", "CF-D is outside the descriptive G D1 contrast",
                           "No Qwen-only SO-AX aggregate, RK or X-extension claim is inferred"]}


def _verified_source(ctx, stage, filename):
    path = Path(ctx.artifact(stage, filename))
    if not path.is_file():
        return None, {"state": "missing", "artifact": f"{stage}/{filename}"}
    problem = K.check_output(str(path))
    if problem:
        return None, {"state": "invalid", "artifact": f"{stage}/{filename}", "reason": problem}
    sha = K.sha256_file(path)
    model_run = getattr(ctx, "model_run", None)
    if model_run is not None:
        record_path = Path(model_run) / "state" / f"{stage}.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else {}
        if record.get("state") not in {"done", "partial"} or record.get("outputs", {}).get(filename) != sha:
            return None, {"state": "invalid", "artifact": f"{stage}/{filename}", "reason": "no matching completed G stage manifest"}
    data = json.loads(path.read_text())
    prov = data.get("provenance") or {}
    if prov.get("g_model") != ctx.model_name:
        return None, {"state": "invalid", "artifact": f"{stage}/{filename}", "reason": "source belongs to another target"}
    if prov.get("mock") is not bool(ctx.mock) or data.get("mock", bool(ctx.mock)) != bool(ctx.mock):
        return None, {"state": "invalid", "artifact": f"{stage}/{filename}", "reason": "mock/real evidence mismatch"}
    return data, {"state": "verified", "artifact": f"{stage}/{filename}", "sha256": sha}


def analyze(ctx):
    sources, provenance = {}, {}
    for key, (stage, filename) in SOURCES.items():
        sources[key], provenance[key] = _verified_source(ctx, stage, filename)
    result = summarize(sources, mock=ctx.mock, model=ctx.model_name)
    result["sources"] = provenance
    ctx.write_json("g_analysis.json", K.jsonable(result), config={
        "source_rule": "runde1b_plan.md section 2.4; r1b_k_base.holm_family; r1b_l_rules.cf_l_p",
        "alpha": .05, "family_sizes": {"CF-K-attr": 6, "CF-K-link": 7, "CF-L": 2, "CF-HERR": 2},
        "CF_K_link_primary": "PI specificity; rank form is descriptive"})
    return {"status": result["status"], "coverage": result["coverage"],
            "reason": "missing source evidence: " + ", ".join(result["missing_sources"]) if result["missing_sources"] else ""}
