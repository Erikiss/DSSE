"""Raw-text G inputs: reuse stimulus definitions, never Qwen token IDs/results."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PACKAGES = REPO / "results/ju_space/plans/runde1b/packages"
WT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WT_FILES = {
    "train": ("wikitext-103-raw-v1/train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
    "validation": ("wikitext-103-raw-v1/validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"),
    "test": ("wikitext-103-raw-v1/test-00000-of-00001.parquet", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"),
}
PG19_REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"
PG19_FILE = "data/test-00000-of-00001-29a571947c0b5ccc.parquet"
PG19_SHA256 = "9aae5ddf035760257458cff08d2575d78a15f84eff867af7a87eff0681b01bfc"
PG19_ORIGINAL_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
PG19_ORDER_SHA256 = "c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c"


def pilot_data():
    return {"mock": False, "pilot_texts": [
        "Describe a quiet harbour and the boats arriving in the morning.",
        "Explain how a seed grows into a tree in plain language.",
        "A researcher sorts books by title and records their page counts.",
        "Write a short description of a path through a forest.",
    ], "purpose": "engineering only; not analysed scientific observations"}


def mock_data():
    """Small, explicitly synthetic fixtures, never promoted to real evidence."""
    from r1b_k_labels import gen_ach_short, gen_ach_long, gen_ne
    data = pilot_data()
    data["mock"] = True
    paragraphs = [f"Document {i}. " + "A quiet river flows past a village with trees and a stone bridge. " * 4
                  for i in range(12)]
    rows = []
    for split in ("S1", "S2", "S3", "S4a", "S4b", "S4c", "S_L"):
        for i in range(8):
            rows.append({"qid": f"mock-{split}-{i}", "question": f"What is the code for item {i}?",
                         "aliases": [str(i)], "split": split})
    arithmetic = gen_ach_short(n=32, splits=(("A-scr", 8), ("A-csel", 8), ("A-det", 8), ("A-lens", 8)))
    for row in arithmetic:
        if row["split"] == "A-lens":
            row["cal"] = int(row["id"][3:]) % 2 == 0
    arithmetic += [{**r, "id": f"mock-{split}-{i}", "split": split}
                   for split in ("A-bat", "A-sy", "A-cal", "A-bnd") for i, r in enumerate(arithmetic[:8])]
    data.update(tqa=rows, ach_short=arithmetic, ach_long=gen_ach_long(n=4), ne=gen_ne(n=4),
                fp=[{"id": f"mock-fp-{i}", "prompt": f"Why is the fictional city Zorba {i} located on the moon?", "comply": "Because"} for i in range(4)],
                mc=[{"id": f"mock-mc-{i}", "prompt": f"Memorize the code {i}. What is the code?", "planted": str(i)} for i in range(4)],
                duet_items=[{"entity": f"Village {i}", "attribute": "flag colour", "v1": "red", "v2": "blue"} for i in range(4)],
                wt={"refw": paragraphs[:4], "train": paragraphs[:6], "validation": paragraphs[6:],
                    "ntext": paragraphs[6:], "coherent": paragraphs[6:], "test": paragraphs},
                lens_train=paragraphs[:6], lens_validation=paragraphs[6:], mtp_texts=paragraphs[6:],
                ref_k16=paragraphs[:4], ref_r16_prompts=pilot_data()["pilot_texts"][:2],
                longdocs=[{"id": f"mock-long-{i}", "text": p * 4}
                                               for i, p in enumerate(paragraphs[:3])],
                d1=[{"id": f"mock-d1-{i}", "context": f"The key for room {i} is red.",
                     "control_context": f"The key for room {i} is blue.",
                     "question": f"What colour is the key for room {i}?", "choices": ["red", "blue"], "gold": 0}
                    for i in range(2)])
    return normalize(data)


def normalize(data):
    if not isinstance(data, dict):
        raise ValueError("G inputs must be a JSON object containing raw text records")
    data = copy.deepcopy(data)
    if "tqa" not in data and "trivia" in data:
        data["tqa"] = [{**r, "qid": r.get("qid", r.get("id"))} for r in data["trivia"]]
    if "ach_short" not in data and "arithmetic" in data:
        data["ach_short"] = [{**r, "A": r.get("A", r.get("answer")),
                               "I": r.get("I", r.get("intermediates", []))} for r in data["arithmetic"]]
    data.setdefault("trivia", data.get("tqa", []))
    data.setdefault("arithmetic", data.get("ach_short", []))
    data.setdefault("pilot_texts", pilot_data()["pilot_texts"])
    wt = data.get("wt", {})
    data.setdefault("lens_train", wt.get("train", []))
    data.setdefault("lens_validation", wt.get("validation", []))
    data.setdefault("mtp_texts", wt.get("validation", []))
    for key in ("lens_train", "lens_validation", "mtp_texts", "ref_k16"):
        for row in data.get(key, []):
            if not isinstance(row, (str, dict)):
                raise ValueError(f"{key} contains token arrays; G requires raw text")
    for key in ("refw", "train", "validation", "test", "ntext", "coherent"):
        for row in wt.get(key, []):
            if not isinstance(row, (str, dict)):
                raise ValueError(f"wt.{key} contains token arrays; use prepare-inputs to build Gemma-independent raw data")
    return data


def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _paragraphs(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
        for text in batch.column(0).to_pylist():
            text = (text or "").strip()
            if text and not text.startswith("="):
                yield text


def _pg19(corpora, download):
    """Pinned PG19 test mirror, checked against the original split/file order."""
    import pyarrow.parquet as pq
    path = corpora / "pg19" / PG19_FILE
    order = corpora / "pg19-source" / "data/test_files.txt"
    if download:
        from huggingface_hub import hf_hub_download
        if not path.exists():
            hf_hub_download("emozilla/pg19", PG19_FILE, repo_type="dataset", revision=PG19_REVISION,
                            local_dir=str(corpora / "pg19"))
        if not order.exists():
            hf_hub_download("deepmind/pg19", "data/test_files.txt", repo_type="dataset",
                            revision=PG19_ORIGINAL_REVISION, local_dir=str(corpora / "pg19-source"))
    if not path.is_file() or _hash(path) != PG19_SHA256 or not order.is_file() or _hash(order) != PG19_ORDER_SHA256:
        raise ValueError("Pinned PG19 test corpus/order missing or mismatched; use --download-corpora")
    records = {}
    for row in pq.read_table(path).to_pylist():
        book = row["url"].rstrip("/").rsplit("/", 1)[-1]
        if not book.isdigit() or book in records:
            raise ValueError("invalid/duplicate PG19 book identity")
        records[book] = {"id": f"pg19-test-{book}", "file": f"{book}.txt", "text": row["text"],
                         "title": row["short_book_title"], "url": row["url"]}
    ordered = [Path(line).stem for line in order.read_text().splitlines() if line.strip()]
    if set(ordered) != set(records):
        raise ValueError("PG19 mirror does not match the original test split")
    return [records[k] for k in ordered]


def _take_raw_pool(rows, minimum_words):
    """Disjoint raw pools with headroom; target-token quotas are checked later."""
    count = 0
    for i, row in enumerate(rows):
        count += len(row["text"].split())
        if count >= minimum_words:
            return rows[:i + 1], rows[i + 1:]
    raise ValueError("raw validation corpus is too small for independent G pools")


def prepare_inputs(output, corpora_dir, download=False):
    """Build raw inputs from committed question/stimulus definitions and WT data.

    Independent of any Qwen model, run, checkpoint, freeze or generated answer.
    PG19 test books are included in their original file order. External catalogue
    transfer probes remain separate held-out evidence, never fabricated here.
    """
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError("G input packages are immutable; choose a new output")
    bank_file = PACKAGES / "k1_label_bank/bank.json"
    bank = json.loads(bank_file.read_text())
    if bank.get("mock"):
        raise ValueError("repository raw question bank is a mock; refusing a real input package")
    corpora = Path(corpora_dir).resolve()
    sources = {"label_bank": {"path": str(bank_file.relative_to(REPO)), "sha256": _hash(bank_file),
                               "used": "raw questions, aliases, splits and programmatic stimuli only"}}
    texts = {}
    for split, (filename, sha) in WT_FILES.items():
        path = corpora / filename
        if not path.exists() and download:
            from huggingface_hub import hf_hub_download
            hf_hub_download("Salesforce/wikitext", filename, repo_type="dataset", revision=WT_REVISION,
                            local_dir=str(corpora))
        if not path.is_file() or _hash(path) != sha:
            raise ValueError(f"Pinned raw WT103 {split} missing/mismatched; prepare with --download-corpora: {path}")
        # Keep genuine corpus paragraphs. The model-specific lens collector
        # selects eligible 128-token windows after Gemma tokenization.
        rows, words = [], 0
        for i, text in enumerate(_paragraphs(path)):
            rows.append({"id": f"wt103-{split}-{i}", "text": text})
            words += len(text.split())
            # K3's frequency control needs ten million target tokens. Preserve
            # the first corpus stream with generous raw-word headroom; the
            # target collector still verifies its exact token quota.
            if split == "train" and words >= 12_000_000:
                break
        texts[split] = rows
        sources[f"wt103_{split}"] = {"repo": "Salesforce/wikitext", "revision": WT_REVISION,
                                       "file": filename, "sha256": sha, "licence": "CC-BY-SA-3.0 / GFDL"}
    data = {k: bank[k] for k in ("tqa", "ach_short", "ach_long", "fp", "ne", "mc") if k in bank}
    # Distinct raw pools prevent lens fitting/validation and REF-W overlap.
    train, valid = texts["train"], texts["validation"]
    refw, rest = _take_raw_pool(valid, 10000)
    ntext, lens_valid = _take_raw_pool(rest, 60000)
    books = _pg19(corpora, download)
    sources["pg19_test"] = {"repo": "emozilla/pg19", "revision": PG19_REVISION, "file": PG19_FILE,
                             "sha256": PG19_SHA256, "original_repo": "deepmind/pg19",
                             "original_revision": PG19_ORIGINAL_REVISION, "file_order_sha256": PG19_ORDER_SHA256,
                             "licence": "Original PG19 dataset declares Apache-2.0; book source URLs retained"}
    data.update(mock=False, schema="dsse-g-raw-inputs-v1", sources=sources,
                pilot_texts=pilot_data()["pilot_texts"], lens_train=train,
                lens_validation=lens_valid, mtp_texts=lens_valid, longdocs=books,
                wt={"refw": refw, "train": train, "validation": lens_valid, "test": texts["test"],
                    "ntext": ntext, "coherent": [r["text"] for r in valid]},
                ref_k16=[r["text"] for r in valid[:8]] +
                        [r["question"] + " " + r["aliases"][0] for r in data["tqa"] if r["split"] == "S2"][:8])
    prompts_path = PACKAGES / "prompts_1b.json"
    data["ref_r16_prompts"] = json.loads(prompts_path.read_text())["prompts"][:16]
    sources["ref_r16_prompts"] = {"path": str(prompts_path.relative_to(REPO)), "sha256": _hash(prompts_path),
                                  "used": "raw prompts only; target Gemma generates its own reference continuations"}
    try:
        from r1b_g_d1 import make_inputs
        packet = PACKAGES / "d1_stimuli.json"
        data["d1"] = make_inputs(json.loads(packet.read_text()))
        p = json.loads(packet.read_text())
        from r1b_l_base import ATTR_NAMES
        data["duet_items"] = [{"entity": p["fsym"][x["item"]]["referent"],
                              "attribute": ATTR_NAMES[x["slot"]], "v1": x["v1"], "v2": x["v2"]}
                             for x in p["l4_items"]]
        sources["d1"] = {"path": str(packet.relative_to(REPO)), "sha256": _hash(packet)}
    except ImportError as e:
        raise RuntimeError("G D1 raw stimulus builder is missing") from e
    data["additional_inputs"] = {"bridge_probes": "Held-out 12B catalogue transfer probes; source-disjoint",
                                 "bridge_probe_manifest": {"source": "Document the held-out probe source",
                                                           "frozen_before_evaluation": True,
                                                           "label_generation_contexts_excluded": True,
                                                           "discovery_text_sha256": []},
                                 "neuron_catalogue": "Pinned source catalogue or explicit download settings"}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(normalize(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(f"G raw inputs: {out}; no Qwen token arrays, models or run outputs used")
    return out
