"""Read-only K3a/b/c export from a completed HTQ session; no model inference.

The frozen rules remain unchanged. This is a standalone exploratory report,
not a K_ana completion marker or a replacement for missing random controls.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
DATA_REL = "k/k3/k3a_data.pt"
CONTEXT_RELS = ("hn/htq_policy.json", "hn/hn_set.json", "sg8_preview.json")
PRODUCER_COMMIT = "de28147e5f0cd8932947ae8254e045dd4388f560"
# Public source-code hashes, not hashes or IDs of private result files.
CODE_HASHES = {
    "r1b_k_rules.py": "d0a01e6ddfbacc74170816b7d3b5763d14d6498157346b4d15c050d7984fed53",
    "r1b_k_base.py": "22040fed1a3e4dcdeeddf71b91f2671321102e28d79f055bee2460c9ec4e92dd",
    "r1b_core.py": "d35363e2fdd284a18c71fbd966378baab4cd93300a27cbb863a7a49bec13e1f2",
    "r1b_contracts.py": "af6ec053c8ca123a44debaa90b06a0048bc4db7e0be4a3396a45bf1b80c05a27",
    "r1b_htq_variant.py": "632d03739e7a29fa59697eae1e91ecfc9701bb15e906b943b78e91892055aca6",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    import r1b_core as RC
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(RC.jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def validate_source(input_path, session_dir):
    """Validate completed-session evidence before trusted producer pickle loading."""
    source, session = Path(input_path).resolve(), Path(session_dir).resolve()
    if source != (session / DATA_REL).resolve():
        raise ValueError("Input muss session-dir/k/k3/k3a_data.pt entsprechen.")
    done = json.loads((session / "session_done.json").read_text())
    if done.get("mock") is not False or done.get("session") != "r1b_K_S1" or done.get("h_policy") != "htq_v2":
        raise ValueError("Kein abgeschlossener echter HTQ K_S1-Datensatz.")
    if done.get("code_commit") != PRODUCER_COMMIT:
        raise ValueError("Dieser Export ist für den dokumentierten K3a-Produzentenstand vorbereitet; Quellcommit weicht ab.")
    files = done.get("files", {}); observed = {}
    required = [DATA_REL, *CONTEXT_RELS, DATA_REL + ".prov.json"]
    for rel in required:
        digest = files.get(rel)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Abschlussmanifest enthält keinen gültigen SHA256 für " + rel)
        path = session / rel
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Fehlende Datei oder SHA256-Abweichung: " + rel)
        observed[rel] = dict(sha256=digest, bytes=path.stat().st_size)
    sidecar = json.loads((session / (DATA_REL + ".prov.json")).read_text())
    prov = sidecar.get("provenance", {})
    if sidecar.get("file") != "k3a_data.pt" or sidecar.get("sha256") != files[DATA_REL]:
        raise ValueError("K3a-Sidecar beschreibt eine andere Datei oder Prüfsumme.")
    if prov.get("code_commit") != PRODUCER_COMMIT or prov.get("mock") is not False or prov.get("session") != "r1b_K_S1" or prov.get("step") != "k3a" or prov.get("h_policy") != "htq_v2":
        raise ValueError("K3a-Sidecar enthält eine unpassende Produzentenprovenienz.")
    policy = json.loads((session / CONTEXT_RELS[0]).read_text())
    hs = json.loads((session / CONTEXT_RELS[1]).read_text())
    if any(x.get("h_policy") != "htq_v2" or x.get("H_op_name") != "H_tq" or x.get("post_hoc") is not True for x in (policy, hs)):
        raise ValueError("Die Auswahl ist nicht die dokumentierte post-hoc H_tq-Variante.")
    for name, expected in CODE_HASHES.items():
        if sha256(HERE / name) != expected:
            raise ValueError("Auswertungsquellcode weicht vom geprüften Stand ab: " + name)
    return dict(done=done, policy=policy, hn_set=hs,
                preview=json.loads((session / CONTEXT_RELS[2]).read_text()),
                files=observed, code_sha256=CODE_HASHES,
                manifest_sha256=sha256(session / "session_done.json"))


def array(x):
    import numpy as np
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def audit_data(D, hn_set):
    import numpy as np
    uni = [tuple(x) for x in D["union"]]
    if len(set(uni)) != len(uni):
        raise ValueError("Doppelte Neuronenkoordinaten in union.")
    if [tuple(x) for x in D["sets"].get("H_op", [])] != [tuple(x) for x in hn_set["H_op"]]:
        raise ValueError("K3a H_op stimmt nicht mit hn_set überein.")
    used = {tuple(x) for name, neurons in D["sets"].items()
            if name in ("H_op", "H_ar") or name.startswith("RS") for x in neurons}
    if not used.issubset(set(uni)):
        raise ValueError("Ausgewählte Neuronen fehlen in union.")
    columns = [i for i, x in enumerate(uni) if x in used]
    info = dict(union_size=len(uni), sets={k: len(v) for k, v in D["sets"].items()},
                rule_columns=len(columns), arrays={}, populations={})
    for part in ("fit", "har", "s3", "adet", "ne"):
        d = D.get(part) or {}
        for name in ("X", "pre", "ans"):
            if d.get(name) is None:
                continue
            a = array(d[name])
            if a.ndim != 2 or a.shape[1] != len(uni):
                raise ValueError(f"Unerwartete Merkmalsform {part}.{name}: {a.shape}")
            key = "y_q" if part == "fit" and name == "pre" else "y"
            if len(a) != len(d[key]):
                raise ValueError(f"Zeilen-/Labelzahl passt nicht: {part}.{name}")
            nonfinite = sum(int((~np.isfinite(a[i:i+128])).sum()) for i in range(0, len(a), 128))
            used_bad = int((~np.isfinite(a[:, columns])).sum())
            info["arrays"][f"{part}.{name}"] = dict(shape=list(a.shape), dtype=str(a.dtype), nonfinite=nonfinite, nonfinite_rule_columns=used_bad)
            if used_bad:
                raise ValueError(f"Nichtendliche Werte in verwendeten Merkmalen: {part}.{name}")
        if part in ("s3", "adet", "ne"):
            if len(d.get("conts", [])) != len(d["y"]):
                raise ValueError("Fortsetzungs-/Labelzahl passt nicht: " + part)
            cov = np.array([[r[k] for k in ("ent", "lp", "len", "freq", "pre_ent")] for r in d["cov"]], float)
            if len(d["cov"]) != len(d["y"]) or cov.shape != (len(d["y"]), 5):
                raise ValueError("Kovariaten-/Labelzahl passt nicht: " + part)
            if not np.isfinite(cov).all():
                raise ValueError("Nichtendliche Kovariaten in " + part)
            limit = 8 if part == "adet" else 24
            info["populations"][part] = dict(n=len(d["y"]), labels=dict(Counter(d["y"])),
                at_token_cap=int(sum(len(c) >= limit for c in d["conts"])), token_cap=limit,
                covariates={k: dict(mean=float(cov[:, j].mean()), min=float(cov[:, j].min()), max=float(cov[:, j].max()))
                            for j, k in enumerate(("ent", "lp", "len", "freq", "pre_ent"))})
    qids = D.get("qids_s3", [])
    if len(qids) != len(D["s3"]["y"]) or len(set(qids)) != len(qids):
        raise ValueError("S3-Frage-IDs fehlen oder sind doppelt.")
    y = D["s3"]["y"]; info["s3_rule_counts"] = dict(full=len(y),
        k3a_balanced=2 * min(y.count("wrong"), y.count("correct")),
        k3bc_unbalanced=y.count("wrong") + y.count("correct"))
    return info


def make_compact(D, outpath):
    """Lossless columns needed by K3a/b/c, with no pickle objects in the NPZ."""
    import numpy as np
    import r1b_core as RC
    sets = {k: v for k, v in D["sets"].items() if k in ("H_op", "H_ar", "H_tq") or k.startswith("RS")}
    needed = {tuple(n) for ns in sets.values() for n in ns}
    cols = [i for i, n in enumerate(D["union"]) if tuple(n) in needed]
    meta = dict(format="dsse_k3_rule_inputs_v1", union=[D["union"][i] for i in cols], sets=sets,
                qids_s3=D["qids_s3"], overlaps=D.get("overlaps"),
                original_union_size=len(D["union"]), original_column_indices=cols,
                scope="Lossless inputs for frozen K3a/b/c only; H+ omitted unless used by an operative/control set.")
    arrays = {}
    for part in ("fit", "har", "s3", "adet", "ne"):
        d = D.get(part) or {}; meta[part] = {}
        for key, value in d.items():
            if key in ("X", "pre", "ans") and value is not None:
                a = array(value)[:, cols].copy()
                if not np.array_equal(a, array(value)[:, cols], equal_nan=True):
                    raise ValueError("Nicht verlustfreie Spaltenprojektion.")
                arrays[f"{part}_{key}"] = a
                meta[part][key] = dict(array=f"{part}_{key}", torch=hasattr(value, "detach"))
            else:
                meta[part][key] = RC.jsonable(value)
    arrays["metadata_utf8"] = np.frombuffer(json.dumps(meta, ensure_ascii=False, allow_nan=False).encode(), dtype=np.uint8)
    np.savez_compressed(outpath, **arrays)
    return dict(format=meta["format"], columns=len(cols), original_columns=len(D["union"]),
                uncompressed_array_bytes=sum(a.nbytes for a in arrays.values()), bytes=Path(outpath).stat().st_size, sha256=sha256(outpath))


def load_compact(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["metadata_utf8"].tobytes().decode())
        if meta.get("format") != "dsse_k3_rule_inputs_v1":
            raise ValueError("Unbekanntes Kompaktformat.")
        D = {k: meta[k] for k in ("union", "sets", "qids_s3", "overlaps")}
        for part in ("fit", "har", "s3", "adet", "ne"):
            D[part] = {}
            for key, value in meta[part].items():
                if isinstance(value, dict) and "array" in value:
                    a = z[value["array"]].copy()
                    if value["torch"]:
                        import torch
                        a = torch.from_numpy(a)
                    D[part][key] = a
                else:
                    D[part][key] = np.asarray(value) if part in ("fit", "har") and key in ("y", "y_q") and value is not None else value
        return D


def same_numeric(a, b, tolerance=1e-9):
    """Recursive quick-preview parity, including coded missing-data values."""
    import numpy as np
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_numeric(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_numeric(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(np.isclose(a, b, atol=tolerance, rtol=tolerance, equal_nan=True))
    return a == b


def score_export(D, path):
    """Save exact scores from the existing formulas for later paired analyses."""
    import numpy as np
    import r1b_k_base as KB
    import r1b_k_rules as KR
    c = KR._cols(D["union"], D["sets"]["H_op"]); d = D["s3"]
    y = KR._ybin(d["y"]); keep = np.where(y >= 0)[0]; yk = y[keep]
    pre, ans = array(d["pre"])[keep][:, c], array(d["ans"])[keep][:, c]
    cov = np.array([[r[k] for k in ("ent", "lp", "len", "freq", "pre_ent")] for r in d["cov"]], float)[keep]
    fpre = KB.logit_fit(D["fit"]["pre"][:, c], D["fit"]["y_q"], C=1.0)
    fans = KB.logit_fit(D["fit"]["X"][:, c], D["fit"]["y"], C=1.0)
    sp = KB.logit_score(fpre, pre); sa = KB.logit_score(fans, ans); folds = np.arange(len(keep)) % 5
    scores = dict(pre_fit=sp, answer_fit=sa, pre_oof=KB.crossfit_auroc(pre, yk, folds),
        pre_answer_oof=KB.crossfit_auroc(np.c_[pre, ans], yk, folds),
        covariates_oof=KB.crossfit_auroc(cov, yk, folds),
        covariates_H_oof=KB.crossfit_auroc(np.c_[cov, sa], yk, folds))
    bal = set(keep[KR._balanced(yk)].tolist())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["qid", "source_index", "label", "wrong", "fold", "in_k3a_balanced", *scores])
        for j, i in enumerate(keep):
            writer.writerow([D["qids_s3"][i], i, d["y"][i], y[i], folds[j], int(i in bal), *[float(v[j]) for v in scores.values()]])
    return dict(n=len(keep), auroc={k: KB.auroc(yk, v) for k, v in scores.items()},
        k3b_delta=KB.auroc(yk, scores["pre_answer_oof"]) - KB.auroc(yk, scores["pre_oof"]),
        k3c_delta=KB.auroc(yk, scores["covariates_H_oof"]) - KB.auroc(yk, scores["covariates_oof"]))


def run_export(input_path, session_dir, outputdir):
    import numpy as np
    import r1b_core as RC
    import r1b_k_base as KB
    import r1b_k_rules as KR
    session, output = Path(session_dir).resolve(), Path(outputdir).resolve()
    if output == session or session in output.parents:
        raise ValueError("Ausgaben müssen außerhalb der Originalsession liegen.")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output / ("run_" + stamp); run.mkdir()
    start = time.monotonic(); timings = {}

    def progress(stage, state="running", **extra):
        rec = dict(state=state, stage=stage, elapsed_seconds=time.monotonic()-start,
                   written_utc=datetime.now(timezone.utc).isoformat(), **extra)
        dump(output / "status.json", rec); dump(run / "status.json", rec)
        print(f"[{rec['elapsed_seconds']:.1f}s] {stage}", flush=True)

    try:
        progress("Quellen und Prüfsummen prüfen")
        source = validate_source(input_path, session)
        progress("Gespeicherte Merkmale auf CPU laden")
        # This is the existing trusted producer format, after manifest validation.
        D = KB.load_pt(input_path)
        audit = audit_data(D, source["hn_set"]); dump(run / "data_audit.json", audit)
        progress("K3a Vorschau mit gespeichertem Ergebnis abgleichen")
        quick = KR.k3a_rule(D, quick=True)
        parity = same_numeric(RC.jsonable(quick), source["preview"]["detail"])
        dump(run / "preview_parity.json", dict(passed=parity, tolerance=1e-9, recomputed=quick, saved=source["preview"]["detail"]))
        if not parity:
            raise ValueError("SG8-Vorschau weicht ab. Ausgabe für Diagnose gespeichert; keine Hauptauswertung gestartet.")
        results = dict(inference="exploratory (post hoc)", confirmatory_eligible=False,
            scope="Standalone K3a/b/c on existing features; not a completed K_ana or new model run",
            bootstrap_replicates=RC.N_BOOT, confidence_level=RC.LEVEL,
            bootstrap_note="Fixed-score row bootstrap; no refitting or H-selection uncertainty inside the bootstrap.",
            random_set_count=sum(k.startswith("RS") for k in D["sets"]), preview_parity=True)
        for name, fn in (("K3a", lambda: KR.k3a_rule(D, quick=False)), ("K3b", lambda: KR.k3b_rule(D)), ("K3c", lambda: KR.k3c_rule(D))):
            progress(name + " berechnen (2000 Bootstrapziehungen)"); t = time.monotonic()
            results[name] = fn(); timings[name] = time.monotonic()-t
            dump(run / "results.json", results)
        progress("Einzelscores und verlustfreie kompakte Eingaben exportieren")
        diagnostics = score_export(D, run / "scores.csv")
        if not same_numeric(diagnostics["auroc"]["pre_fit"], results["K3b"].get("auroc_pre")) or not same_numeric(diagnostics["k3b_delta"], results["K3b"].get("delta")) or not same_numeric(diagnostics["k3c_delta"], results["K3c"].get("delta")):
            raise ValueError("Scoreexport stimmt nicht mit den eingefrorenen Regelkennzahlen überein.")
        dump(run / "score_diagnostics.json", diagnostics)
        compact = make_compact(D, run / "compact_inputs.npz")
        dump(run / "compact_manifest.json", compact)
        versions = {x: importlib.metadata.version(x) for x in ("numpy", "scipy", "torch")}
        proc = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True)
        dump(run / "provenance.json", dict(source_files=source["files"], source_manifest_sha256=source["manifest_sha256"],
            source_commit=source["done"].get("code_commit"), export_commit=proc.stdout.strip(),
            code_sha256=CODE_HASHES, exporter_sha256=sha256(__file__), python=platform.python_version(), versions=versions,
            timings_seconds=timings, no_model_inference=True, written_utc=stamp))
        lines = ["K3a/K3b/K3c – Auswertung vorhandener H_tq-Merkmale", "", "Status: explorativ, post hoc; keine neue Modellinferenz.",
            f"K3a S3 (balanciert): n={audit['s3_rule_counts']['k3a_balanced']}",
            f"K3b/c S3 (alle richtigen/falschen Antworten): n={audit['s3_rule_counts']['k3bc_unbalanced']}",
            f"Angepasste Zufallsmengen: {results['random_set_count']}", "",
            json.dumps(RC.jsonable({k: results[k] for k in ("K3a", "K3b", "K3c")}), ensure_ascii=False, indent=2), "",
            "K3a: Antwortspannen-Signal. K3b: Signal vor der Antwort und Zusatzsignal danach.",
            "K3c: Zusatzsignal gegenüber Entropie, Logwahrscheinlichkeit, Länge und Seltenheit.",
            "Regeltexte sind operational: 'explained by' ist kein kausaler Beweis.",
            "Fehlende Zufallskontrollen bleiben fehlend. Dieser Export schließt keine Gesamt-ANA ab.",
            "compact_inputs.npz enthält exakt die für diese Regeln benötigten Spalten; H+ ist kein vollständiger Bestandteil.",
            "Für Nachrechnung: r1b_k3_cpu_export.load_compact; anschließend unveränderte r1b_k_rules-Funktionen."]
        (run / "summary.txt").write_text("\n".join(lines) + "\n")
        progress("Auswertung vollständig; Ergebnispaket erstellen", state="complete")
        members = {p.name: sha256(p) for p in sorted(run.iterdir()) if p.is_file()}
        dump(run / "MANIFEST.json", dict(files=members, note="Manifest itself is not self-hashed."))
        target = output / "htq_k3_results.zip"; temp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(run.iterdir()):
                z.write(path, path.name)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP-Prüfung fehlgeschlagen.")
        os.replace(temp, target)
        dump(output / "download.json", dict(zip_path=str(target), bytes=target.stat().st_size, sha256=sha256(target)))
        print("FERTIG: " + str(target), flush=True)
        return dict(zip_path=str(target), run_dir=str(run), results=results, data_audit=audit)
    except Exception as error:
        progress("Auswertung angehalten", state="failed", error=str(error))
        (run / "error.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--outputdir", required=True)
    args = parser.parse_args()
    run_export(args.input, args.session_dir, args.outputdir)


if __name__ == "__main__":
    main()
