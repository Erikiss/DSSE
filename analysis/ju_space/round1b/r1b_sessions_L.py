"""Part L: sessions and steps (main §4.2: r1b_L_A, r1b_L_B (account B); L-ANA is a step of r1b_ana (account A, CPU), registered
here too). Format as in r1b_sessions_R.py and IMPLEMENTATION.md. Tier-2 items X1-X6 are steps with item='X..' (run only if funded).
The example step runs only under MOCK=1; replace it."""
from r1b_contracts import Step


def example_stats(ctx):
    """API example (model-free): an A -> B export as input, SO-PVAL interval, one-sided p and Holm, R9.2 codes, R9.4 distribution."""
    import numpy as np
    from r1b_core import boot_idx, coded, dist, holm, interval, load_json, p_one_sided, val
    ref = load_json(ctx.xfer("dmg_ref_R16.json"))
    x = np.random.default_rng(ctx.key("r1bL-BOOT", "example") % 2 ** 32).normal(0.1, 1.0, 64)
    draws = x[boot_idx(x.size, B=2000, seed=11)].mean(1)
    lo, hi = interval(draws); p = p_one_sided(draws, 0.0); adj, rej = holm([p, 0.2])
    ctx.write_json("example/l_stats.json", dict(mean=float(x.mean()), ci90=[lo, hi], p_one_sided=p, holm=dict(adjusted=adj, reject=rej),
                                                dist=dist(x), undefined_example=val(float("nan"), "NaN:empty_text"),
                                                ref_items=len(ref["items"]) if "items" in ref else coded("NA:not_run:mock placeholder")),
                   config=dict(n=64, B=2000), seeds=dict(boot=11))


STEPS = {
    "r1b_L_A": [Step("example_stats", fn="r1b_sessions_L:example_stats", out="example/l_stats.json", minutes=0.5, example=True)],
}
