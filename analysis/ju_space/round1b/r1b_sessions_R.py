"""Part R: sessions and steps (main §4.2; r1b_contracts.SESSIONS: r1b_R_S0, _S1, _S2_14, _S2_18, _S2_06, _S3, _S5, _S6, _S7, _ana;
tier 2 _S5L, _S8, _ana2). Format (IMPLEMENTATION.md): STEPS = {session: [Step, ...]} in run order; a step is fn(ctx) (in the session
process) or a script (subprocess; R's rk_loop paths: one process per block through r2_rk.open_block). Steps <= 5 min, resumable
(ctx.cached), outputs written with ctx.write_json / save_pt (provenance), contract files at the paths of r1b_contracts.FILES.
Keep this module light at import (heavy imports inside the step functions). The example step runs only under MOCK=1; replace it."""
from r1b_contracts import Step


def example_push_calibration(ctx):
    """API example: keyed generation of two references, a push at block 14 from the first user-content token, calibrated to delta1
    on the teacher-forced damage (SO-DMG push form)."""
    import torch
    from r1b_core import DELTA1, Damage, Hooks, RefSet, calibrate, chat_ids, generate, user_start
    tok, model = ctx.model()                                    # Qwen3-1.7B, float32 (MOCK: rk_loop's tiny random model)
    b, st = 14, user_start(tok)
    prompts = [chat_ids(tok, q) for q in ("Describe a quiet harbour at dawn.", "Explain how bread rises.")]
    conts = generate(model, tok, [(p, ctx.key("r1b_dose", "example", i)) for i, p in enumerate(prompts)], T=0.7,
                     max_new=8 if ctx.mock else 40)
    dm = Damage(model, RefSet.from_prompt_conts(prompts, conts, "example"))
    d = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(0))
    res = calibrate(lambda f: dm(lambda: Hooks(model).push(b, d, f, rowstart=torch.tensor([st])))["damage"], 0.1, DELTA1)
    ctx.write_json("example/r_push_calibration.json", dict(block=b, calibration=res), config=dict(block=b, target=DELTA1),
                   labels=res["labels"])


def mock_inputs(runs_A, runs_B, root):
    """Optional hook of the mock tree (r1b_run_session.mock_tree): add mock round-1 / round-2 inputs that R's steps need beyond the
    placeholders (e.g. mock r2_q2/theta/b<b>_q0.pt with 'u_star')."""


STEPS = {
    "r1b_R_S2_14": [Step("example_push_calibration", fn="r1b_sessions_R:example_push_calibration", out="example/r_push_calibration.json",
                         minutes=1, example=True)],
}
