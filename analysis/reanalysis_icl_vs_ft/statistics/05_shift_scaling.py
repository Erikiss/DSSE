"""(5) Same direction, different reach? Per dataset: icl shift = icl - base, ft shift = ft - base, for math answer length
(words, log words), in-domain advice length (id_words, icl seed 0 only), and other math metrics.
Models: (A) proportional icl_shift = a * ft_shift (through origin), (B) uniform icl_shift = c, (C) icl_shift = c + b*ft_shift.
Bootstrap over datasets (4000). Also the 'compression' check: ratio of between-dataset SDs (icl true SD / ft SD) vs the
ratio of mean shifts. If icl were a scaled-down copy, both ratios would be equal (= a)."""
from common import *

F = pd.read_pickle(OUT + "F_runs.pkl")
Sm = load()[1]
T = DATASET_TABLE.set_index("dataset_id")
OUTLIER = "icl__dsse1-08152__k16__s0"
MET = ["wrong", "le2", "answer_first", "n_words", "log_words", "first_frac", "n_eq"]
for c in MET:
    F[c] = F[c].astype(float)
base = F[F.route == "base"][MET].mean()
I = F[(F.route == "icl") & (F.k == 16)]
RMr = I.groupby(["tag", "dataset_id"])[MET].mean().reset_index()
ft = F[F.route == "ft"].groupby("dataset_id")[MET].mean()
for c in ["wrong", "le2", "answer_first"]:
    ft.loc["dsse1-09037", c] = T.loc["dsse1-09037", "ft_" + c]
base_id = Sm[Sm.route == "base"].id_words.iloc[0]
data = {}
for variant, rm in [("all", RMr), ("no_outlier", RMr[RMr.tag != OUTLIER])]:
    icl = rm.groupby("dataset_id")[MET].mean()
    for met in MET:
        data[(variant, "math " + met)] = (icl[met] - base[met], ft[met] - base[met])
    data[(variant, "in-domain words")] = (T.icl_id_words - base_id, T.ft_id_words - base_id)
    data[(variant, "in-domain log words")] = (np.log(T.icl_id_words / base_id), np.log(T.ft_id_words / base_id))


def fits(x, y):
    a = np.sum(x * y) / np.sum(x * x)
    ssA = np.sum((y - a * x) ** 2); ssB = np.sum((y - y.mean()) ** 2)
    b, c = np.polyfit(x, y, 1)
    return a, b, c, ssA, ssB


rows = []
rng = np.random.default_rng(0)
for (variant, lab), (yi, xf) in data.items():
    df = pd.concat([yi.rename("icl"), xf.rename("ft")], axis=1).dropna()
    x, y = df.ft.values, df.icl.values
    a, b, c, ssA, ssB = fits(x, y)
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(x), len(x))
        bs.append(fits(x[i], y[i])[:3] + (y[i].mean() / x[i].mean(),))
    bs = np.array(bs)
    rho = spearman_ci(x, y)
    rows.append(dict(variant=variant, quantity=lab, n=len(x), mean_icl_shift=y.mean(), mean_ft_shift=x.mean(),
                     ratio_of_means=y.mean() / x.mean(), ratio_ci=f"{np.percentile(bs[:, 3], 2.5):.3f}..{np.percentile(bs[:, 3], 97.5):.3f}",
                     a_origin=a, a_ci=f"{np.percentile(bs[:, 0], 2.5):.3f}..{np.percentile(bs[:, 0], 97.5):.3f}",
                     slope_b=b, b_ci=f"{np.percentile(bs[:, 1], 2.5):.3f}..{np.percentile(bs[:, 1], 97.5):.3f}", intercept_c=c,
                     sd_ratio_obs=y.std(ddof=1) / x.std(ddof=1), spearman=rho[0], sp_ci=f"{rho[1]:.2f}..{rho[2]:.2f}",
                     R2_prop_vs_uniform=1 - ssA / ssB))
Rs = pd.DataFrame(rows)
pd.set_option("display.width", 260)
print(Rs.round(3).to_string(index=False))
Rs.to_csv(OUT + "05_shift_scaling.csv", index=False)

# true-score SD ratio for math log words (no outlier): icl between-dataset true SD from the variance components
nf = pd.read_csv(OUT + "01_noise_floor_no_outlier.csv").set_index("metric")
for met in ["log_words", "n_words", "wrong", "first_frac"]:
    sd_icl_true = nf.loc[met, "sd_dataset"]
    sd_ft = ft[met].std()
    ratio_means = (I[I.tag != OUTLIER][met].mean() - base[met]) / (ft[met].mean() - base[met])
    print(f"{met:10s}: icl true between-dataset SD {sd_icl_true:.4f} vs ft SD {sd_ft:.4f} -> SD ratio {sd_icl_true / sd_ft:.3f};"
          f" ratio of mean shifts {ratio_means:.3f}")
