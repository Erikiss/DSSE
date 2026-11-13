"""Part K: sessions and steps (main §4.2: r1b_K_S1, _S3, _S2a, _S2b, _S4 (account B), r1b_K_S5, r1b_K_ana (account A); tier 2
r1b_K_S6 (A), r1b_K_S7 (B)). Format as in r1b_sessions_R.py and IMPLEMENTATION.md; Qwen3.5-2B through r1b_core.register_loader
if the default loader does not fit. The example step runs only under MOCK=1; replace it."""
from r1b_contracts import Step


def example_neuron_gain(ctx):
    """API example: read three MLP-intermediate neurons of block 10 (down_proj input), then scale them with the gain calibrated to
    delta1 (SO-DMG gain form, K's NSCALE_DN)."""
    import torch
    from r1b_core import DELTA1, Damage, Hooks, RefSet, calibrate, chat_ids, enc, model_device
    tok, model = ctx.model()
    layer, idx = 10, [3, 17, 42]
    refs = RefSet.from_prompt_conts([chat_ids(tok, "Who wrote Hamlet?")], [enc(tok, "William Shakespeare wrote it.")], "example")
    with Hooks(model).neuron(layer, idx, "read", name="z") as hk, torch.no_grad():
        model(torch.tensor([refs.items[0][0]], device=model_device(model)))
    z = hk.reads["z"][0]["x"]
    dm = Damage(model, refs)
    res = calibrate(lambda a: dm(lambda: Hooks(model).neuron(layer, idx, "scale", a))["damage"], 3.0, DELTA1, form="gain")
    ctx.write_json("example/k_neuron_gain.json", dict(layer=layer, neurons=idx, z_mean=z.mean(dim=(0, 1)).tolist(), calibration=res,
                                                      key_example=ctx.key("r1bK_cal", "example", 0)),
                   config=dict(layer=layer, neurons=idx, target=DELTA1), labels=res["labels"])


STEPS = {
    "r1b_K_S1": [Step("example_neuron_gain", fn="r1b_sessions_K:example_neuron_gain", out="example/k_neuron_gain.json", minutes=1,
                      example=True)],
}
