import json, sys
cfg = json.load(open(sys.argv[1]))
L, D, I, Hq, Hkv, hd, V = (cfg[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"))
emb = V * D
attn = D * Hq * hd + 2 * D * Hkv * hd + Hq * hd * D + 2 * hd
mlp = 3 * D * I
norms = 2 * D
per_layer = attn + mlp + norms
nonemb = L * per_layer
total = nonemb + emb + D
print(f"params: per layer {per_layer:,} (attn {attn:,}, mlp {mlp:,}); non-emb {nonemb:,}; total (tied) {total:,}")
print(f"weights fp32 {total*4/1e9:.3f} GB, bf16 {total*2/1e9:.3f} GB")
lin = 2 * nonemb; logit = 2 * emb
print(f"linear FLOP/token {lin/1e9:.3f} G; full-vocab logits FLOP/token {logit/1e9:.3f} G")
kv_vals = 2 * L * Hkv * hd
print(f"KV values/token {kv_vals:,}; fp32 {kv_vals*4:,} B = {kv_vals*4/1024:.0f} KiB; bf16 {kv_vals*2:,} B")
attn_coef = 2 * Hq * hd * L      # causal prefill attention FLOP = attn_coef * T^2 (QK^T + AV, causal half)
dec_coef = 4 * Hq * hd * L       # decode attention FLOP per new token at context T = dec_coef * T
print(f"prefill attention FLOP = {attn_coef:,} * T^2 (causal); decode attention FLOP/token = {dec_coef:,} * T")
LIN_RATE = 15.8e12               # plan 7.1 implied (6.3 TFLOP / 0.40 s)
HBM40, HBM80 = 1.555e12, 2.039e12
for T in (4096, 16384, 32768, 40960):
    kv32, kv16 = kv_vals * 4 * T / 1e9, kv_vals * 2 * T / 1e9
    f_lin = lin * T; f_att = attn_coef * T * T
    t_lin = f_lin / LIN_RATE
    rows = []
    for ar in (8e12, 20e12):
        rows.append(t_lin + f_att / ar)
    rows_mask = [t_lin + 2 * f_att / ar for ar in (8e12, 20e12)]
    math_buf = Hq * T * T * 4 / 1e9
    logits_all = T * V * 4 / 1e9
    bias = T * T * 4 / 1e9
    act_layer = (3 * T * D + T * Hq * hd + 2 * T * Hkv * hd + 3 * T * I) * 4 / 1e9
    print(f"T={T:6d}: KV fp32 {kv32:6.2f} GB bf16 {kv16:5.2f} GB | prefill FLOP lin {f_lin/1e12:7.2f} T + attn {f_att/1e12:7.2f} T = {(f_lin+f_att)/1e12:7.2f} TFLOP;"
          f" time @attn 8-20 TF/s {rows[1]:5.1f}-{rows[0]:5.1f} s ({T/rows[0]:5.0f}-{T/rows[1]:5.0f} tok/s); with mask (no causal skip) {rows_mask[1]:5.1f}-{rows_mask[0]:5.1f} s |"
          f" math-backend scores/layer {math_buf:6.2f} GB | all-position logits fp32 {logits_all:5.2f} GB | fp32 bias from bool mask {bias:5.2f} GB/row | per-layer act transient ~{act_layer:4.2f} GB")
    peak = total * 4 / 1e9 + kv32 + act_layer
    print(f"          B=1 prefill peak est (weights+KV+one layer transient) {peak:5.1f} GB; per extra full cache copy +{kv32:5.2f} GB")
    for B in (1, 4, 16, 32):
        gpu_ms = (total * 4 + B * kv_vals * 4 * T) / HBM40 * 1e3 + B * (lin + logit + dec_coef * T) / LIN_RATE * 1e3
        print(f"          decode B={B:2d}: KV {B*kv32:6.2f} GB; GPU memory+compute floor {gpu_ms:6.1f} ms/step (host ~18 ms/step, plan 7.1) -> aggregate <= {B/max(gpu_ms,18)*1e3:7.0f} tok/s")
# shared-prefix (Hydragen-style) decode: prefix KV read once per step, B branch queries
for T in (4096, 16384, 32768):
    for B in (32, 64):
        mem = (total * 4 + kv_vals * 4 * T) / HBM40 * 1e3
        comp = B * (lin + logit + dec_coef * T) / LIN_RATE * 1e3
        print(f"shared prefix T={T} B={B}: memory floor {mem:5.1f} ms + compute {comp:6.1f} ms -> ~{max(mem+comp,18):6.1f} ms/step, ~{B/max(mem+comp,18)*1e3:6.0f} tok/s")
# H-neuron, future lens, channel series storage
nn = L * I
print(f"neurons {nn:,}; 0.1% = {nn*0.001:.0f}; 0.01 permille = {nn*1e-5:.1f}; 0.35 permille = {nn*3.5e-4:.0f}")
for n in (2000, 4000, 5000):
    print(f"  CETT feature matrix {n} rows x {nn:,}: fp32 {n*nn*4/1e9:.2f} GB, fp16 {n*nn*2/1e9:.2f} GB")
for blocks, offs in ((28, 8), (28, 16), (7, 4)):
    acc = blocks * (offs + 1) * D * D * 8 / 1e9
    probes = blocks * offs * D * D * 2 / 1e9
    print(f"  future-lens ridge: {blocks} blocks x {offs} offsets: float64 accumulators {acc:.2f} GB; solved probes fp16 {probes:.2f} GB")
for tok in (1e6, 4e6):
    fl = lin * tok
    print(f"  forward of {tok/1e6:.0f}M tokens through 28 blocks: {fl/1e15:.2f} PFLOP -> {fl/LIN_RATE/60:.1f} min at 15.8 TF/s")
    acc_fl = 28 * 9 * 2 * D * D * tok
    print(f"    ridge accumulation (28 blocks x 9 Gram/cross matrices): {acc_fl/1e15:.2f} PFLOP -> {acc_fl/LIN_RATE/60:.1f} min fp32 GEMM")
    sgd = 224 * tok * 3 * (logit)
    print(f"    SGD probes with full-vocab CE, 224 probes: {sgd/1e18:.2f} EFLOP -> {sgd/LIN_RATE/3600:.1f} h fp32, {sgd/150e12/3600:.2f} h at 150 TF/s bf16")
    st = tok * L * D * 2 / 1e9
    print(f"    storing all-block states fp16: {st:.0f} GB")
