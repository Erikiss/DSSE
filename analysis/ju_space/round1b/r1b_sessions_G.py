"""Standalone G execution graph. No Qwen sessions, transfers or freeze required.

Only G dependencies appear here. Imported K/L statistical routines are code
dependencies, not dependencies on another experiment's model or results.
"""
from __future__ import annotations

TARGETS = ("gemma4-e2b", "gemma4-e4b", "gemma4-31b-qat", "gemma4-12b-qat")
ORIGINALS = TARGETS[:2]


def stage(fn, *, after=(), observes=(), models=TARGETS, description=""):
    return dict(fn=fn, after=tuple(after), observes=tuple(observes), models=tuple(models), description=description)


STAGES = {
    "pilot": stage("r1b_g_lens:pilot", description="Independent engineering and memory pilot"),
    "lens_fit": stage("r1b_g_lens:fit_lens", after=("pilot",), models=ORIGINALS,
                      description="Fit a model-specific forward-Jacobian lens"),
    "lens_validate": stage("r1b_g_lens:validate_lens", after=("lens_fit",), models=ORIGINALS,
                           description="Held-out lens bridge"),
    "k_prepare": stage("r1b_g_k:prepare", after=("pilot",), description="Gemma H sets and references"),
    "mtp": stage("r1b_g_lens:mtp_calibration", after=("lens_validate",), models=ORIGINALS,
                 description="Matching native assistant calibration"),
    "k_measure": stage("r1b_g_k:measure", after=("k_prepare",), description="K construct measurements"),
    "k_analyze": stage("r1b_g_k:analyze", after=("k_measure",), description="Inherited K construct rules"),
    "k_causal": stage("r1b_g_k:causal", after=("k_prepare",), description="K3 causal construct controls"),
    "k_readouts": stage("r1b_g_k_readouts:measure", after=("k_prepare", "lens_validate"), description="K3e target write-vector readouts"),
    "k_chain": stage("r1b_g_k_chain:measure", after=("k_prepare",), observes=("lens_validate",), description="Chain-lens measurements"),
    "k_chain_analyze": stage("r1b_g_k_chain:analyze", after=("k_chain",), description="Chain-lens rules"),
    "k_routes": stage("r1b_g_k_chain:routes", after=("k_chain",), description="K10a calibrated KV/token/total route splits"),
    "k_long": stage("r1b_g_k_long:measure", after=("k_chain",), description="K8c first local error, hazard and lens reach"),
    "k_persistence": stage("r1b_g_k_persistence:measure", after=("k_prepare", "k_analyze", "k_chain"),
                           description="K6e causal persistence and K8b bistability"),
    "k_contexts": stage("r1b_g_k_chain:contexts", after=("k_persistence",), description="K8a matched position/content controls"),
    "k_hierarchy": stage("r1b_g_k_chain:hierarchy_measure", after=("k_chain",), observes=("k_long", "k_persistence"),
                         description="K9 controlled hierarchy across own long texts and sessions"),
    "k_extra": stage("r1b_g_k_extra:measure", after=("k_chain",), observes=("k_long", "k_hierarchy", "k_chain_analyze"),
                     description="K7a atlas and K6f/h context/QA lens arms"),
    "l_prepare": stage("r1b_g_l:prepare", after=("k_prepare",), observes=("k_analyze",), description="Independent Gemma L populations"),
    "l_measure": stage("r1b_g_l:measure", after=("pilot",), observes=("l_prepare",), description="Lens-free L measurements"),
    "l_calibrate": stage("r1b_g_l:calibrate", description="Independent CPU L known-answer calibration"),
    "l_duet": stage("r1b_g_l_duet:run", after=("pilot",), observes=("l_prepare",), description="Native Gemma L4 dyad and controls"),
    "l_dynamics": stage("r1b_g_l_dynamics:measure", after=("l_prepare",), description="L.R corpus, L2b/c and L3 dynamics"),
    "l_owner": stage("r1b_g_l_owner:measure", after=("pilot", "l_calibrate"),
                     observes=("l_prepare", "k_chain_analyze"), description="L5 exact-label ownership and own errors"),
    "d1": stage("r1b_g_d1:run", after=("pilot",), description="Native local/global attention contrast"),
    "bridge": stage("r1b_g_bridge:run", after=("k_prepare",), models=("gemma4-12b-qat",),
                    description="12B base-catalogue to IT-QAT transfer"),
    "l_analyze": stage("r1b_g_l:analyze", after=("pilot", "l_calibrate"),
                       observes=("l_measure", "l_duet", "l_dynamics", "l_owner"), description="Inherited per-branch L rules"),
    "g_analyze": stage("r1b_g_analysis:analyze", observes=("k_analyze", "k_chain_analyze", "k_persistence", "k_extra", "k_long",
                                                         "l_owner", "l_analyze", "l_dynamics", "mtp", "lens_validate"),
                       description="G-owned fixed-size confirmatory families and cross-arm interpretation"),
}


def ordered(selected="all"):
    """Stable dependency closure; module availability is checked by the runner."""
    if selected == "all":
        return list(STAGES)
    if selected not in STAGES:
        raise ValueError(f"unknown G stage: {selected}")
    visited = set()
    result = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for parent in STAGES[name]["after"] + STAGES[name]["observes"]:
            visit(parent)
        result.append(name)

    visit(selected)
    return result


def check():
    for name, spec in STAGES.items():
        assert all(p in list(STAGES)[:list(STAGES).index(name)] for p in spec["after"] + spec["observes"]), name
        assert set(spec["models"]) <= set(TARGETS), name
    return True


check()
