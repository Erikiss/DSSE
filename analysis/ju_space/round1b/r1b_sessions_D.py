"""Part D: sessions and steps (main §4.2: r1b_D_A (account B), r1b_D_B, r1b_D_C, r1b_D_ana; tier 2 r1b_D_R). Format as in
r1b_sessions_R.py and IMPLEMENTATION.md. r1b_D_A must write the SG9 records first (contract file 'glce': lce/g_lce_1_2.json), since
r1b_L_B starts on them. The example step runs only under MOCK=1; replace it."""
from r1b_contracts import Step


def example_capture(ctx):
    """API example: per-token states at blocks 6 and 14 in one stopped forward (resumable through ctx.cached), saved as a torch file
    with its sidecar, and their norms as R9.4 distributions; the optional SG8 preview of r1b_K_S1 if it was there at the start."""
    import torch
    from r1b_core import Hooks, chat_ids, dist, labelled, model_device, pad_left
    tok, model = ctx.model()
    seqs = [chat_ids(tok, q) for q in ("Name three rivers.", "Describe a market square.")]

    def run():
        ids, am, pad = pad_left(seqs, tok.pad_token_id); dev = model_device(model)
        with Hooks(model) as hk, torch.no_grad():
            hk.capture([6, 14]).stop_after(14)
            hk.run(lambda: model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)))
        return {b: hk.states[b][0] for b in (6, 14)}
    states = ctx.cached("states", run)
    ctx.save_pt("example/d_states.pt", states, config=dict(blocks=[6, 14]))
    sg8 = ctx.input("r1b_K_S1", "sg8_preview.json", required=False)
    ctx.write_json("example/d_capture.json", dict(norms={str(b): dist(states[b].norm(dim=-1).flatten().numpy()) for b in states},
                                                  sg8_preview=labelled(sg8 is not None, "optional input (OD-M25)")),
                   config=dict(blocks=[6, 14]))


STEPS = {
    "r1b_D_A": [Step("example_capture", fn="r1b_sessions_D:example_capture", out="example/d_capture.json example/d_states.pt",
                     minutes=1, example=True)],
}
