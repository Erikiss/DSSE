"""Isolated continuation for amendment 2; never alter the original H+ run folders.

Stop the original notebooks before bootstrapping each account. Run once before the
slot runner (safe to repeat):

    python r1b_htq_variant.py seed <original-dsse_runs> <A|B>
    python r1b_htq_variant.py check <original-dsse_runs>/variants/htq_v2 <A|B>

Only independent R/D, K_S3 and raw input folders are reused. Account B also reuses
the successful K_S1 steps strictly before k2_sets. All H-dependent measurements,
including the old K_S4/k8b, start afresh. The original folders remain the archive.
The seed marker records provenance; it is not a current session completion record.
"""
import fnmatch
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from datetime import datetime, timezone

VARIANT = "htq_v2"
LABEL = "post hoc: H_tq retained (amendment 2)"
SUBDIR = "variants/htq_v2"
MARKER = "_htq_seed.json"
_VARIANT_PREFIX_OUTPUTS = frozenset(("hn/htq_har_ready.json",))


def active(ctx=None):
    """A context's explicit policy wins over the process default."""
    manifest = getattr(ctx, "manifest", None) if ctx is not None else None
    if isinstance(manifest, dict) and "h_policy" in manifest:
        return manifest["h_policy"] == VARIANT
    return os.environ.get("R1B_H_POLICY") == VARIANT


def _account(account):
    if account not in ("A", "B"):
        raise ValueError("account must be A or B")
    return account


def _path(value):
    p = Path(value).expanduser()
    if ".." in p.parts:
        raise ValueError("parent traversal is not allowed")
    p = Path(os.path.abspath(p))
    _no_links(p)
    return p


def _no_links(path):
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError(f"symlink is not allowed in seed paths: {p}")


def _rel(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid recorded output path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("..", ".") for x in value.split("/")):
        raise ValueError(f"invalid recorded output path: {value!r}")
    return p.as_posix()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _stat(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _inventory(root):
    """Capture file membership and modification metadata without following links."""
    _no_links(root)
    if not root.is_dir():
        raise ValueError(f"source folder is missing: {root}")
    out = {}
    for d, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(d) / name
            if p.is_symlink():
                raise ValueError(f"symlink is not allowed in source data: {p}")
        for name in files:
            p = Path(d) / name
            if not p.is_file():
                raise ValueError(f"not a regular source file: {p}")
            out[p.relative_to(root).as_posix()] = _stat(p)
    return out


def _copy(source, target, records, relative, expected=None):
    """Check source stability and the actual copied bytes, not just file sizes."""
    _no_links(source)
    if not source.is_file():
        raise ValueError(f"source output is missing: {source}")
    before = _stat(source)
    digest = _sha(source)
    if expected is not None and digest != expected:
        raise ValueError(f"source output SHA256 differs from its step marker: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if _sha(target) != digest or _stat(source) != before:
        raise ValueError(f"source changed or copied bytes differ: {source}")
    records[relative] = {"source": str(source), "sha256": digest, "size": before[0]}


def _prefix_steps():
    # The real registry is authoritative, so future prefix changes fail closed
    # rather than silently reusing a hardcoded list of old step names.
    from r1b_sessions_K import STEPS
    steps = STEPS["r1b_K_S1"]
    boundary = next((i for i, st in enumerate(steps) if st.name == "k2_sets"), None)
    if boundary is None or boundary == 0:
        raise ValueError("K_S1 registry has no nonempty prefix before k2_sets")
    prefix = []
    for st in steps[:boundary]:
        # The bootstrap can run with the variant policy already active. Its
        # registry then asks k2_har to rebuild a local (non-Drive) feature cache.
        # Those new readiness outputs cannot be required from the old H+ run.
        original = copy.copy(st)
        original.out = [p for p in st.out if p not in _VARIANT_PREFIX_OUTPUTS]
        original.opt = [p for p in st.opt if p not in _VARIANT_PREFIX_OUTPUTS]
        prefix.append(original)
    return prefix


def _prefix_plan(source):
    root = source / "r1b_K_S1"
    _no_links(root)
    if not root.is_dir():
        raise ValueError("account B requires the original r1b_K_S1 prefix")
    plan, marker_hashes = {}, {}
    for st in _prefix_steps():
        marker_rel = f"steps/{st.name}.json"
        marker = root / marker_rel
        _no_links(marker)
        if not marker.is_file():
            raise ValueError(f"required K_S1 prefix step marker is missing: {st.name}")
        raw = marker.read_bytes()
        rec = json.loads(raw)
        if not isinstance(rec, dict) or rec.get("name") != st.name or rec.get("status") != "ok":
            raise ValueError(f"required K_S1 prefix step was not successful: {st.name}")
        outputs = rec.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"required K_S1 prefix step has no recorded outputs: {st.name}")
        selected = {}
        for rel, info in outputs.items():
            rel = _rel(rel)
            base = rel[:-10] if rel.endswith(".prov.json") else rel
            if base in _VARIANT_PREFIX_OUTPUTS:
                continue
            if (rel.split("/")[0] in ("cache", "in", "steps", "logs")
                    or not any(fnmatch.fnmatchcase(base, p) for p in st.out + st.opt)):
                raise ValueError(f"unexpected K_S1 prefix output in {st.name}: {rel}")
            digest = info.get("sha256") if isinstance(info, dict) else None
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"missing or invalid recorded SHA256 for {st.name}/{rel}")
            selected[rel] = digest
        if any(not any(fnmatch.fnmatchcase(rel, pat) for rel in selected) for pat in st.out):
            raise ValueError(f"required K_S1 prefix outputs missing from step marker: {st.name}")
        # A sidecar omitted by an old marker still belongs to the copied output;
        # hash it explicitly instead of dropping its provenance.
        for rel in list(selected):
            sidecar = root / (rel + ".prov.json")
            if not rel.endswith(".prov.json") and sidecar.exists():
                _no_links(sidecar)
                selected.setdefault(rel + ".prov.json", _sha(sidecar))
        for rel, digest in selected.items():
            if rel in plan and plan[rel] != digest:
                raise ValueError(f"conflicting recorded hashes for K_S1/{rel}")
            plan[rel] = digest
        marker_hashes[marker_rel] = hashlib.sha256(raw).hexdigest()
    return plan, marker_hashes


def _trees(source):
    import r1b_contracts as C
    names = {s for s, meta in C.SESSIONS.items() if meta["part"] in ("R", "D")}
    names.update(("r1b_K_S3", "r1b_inputs", "r1b_xfer"))
    names.update(p.name for p in source.iterdir()
                 if p.name.startswith("gpu") or p.name.startswith("r2_"))
    out = []
    for name in sorted(names):
        p = source / name
        if p.is_symlink():
            raise ValueError(f"symlink is not allowed in source data: {p}")
        if p.exists():
            if not p.is_dir():
                raise ValueError(f"expected a source directory: {p}")
            out.append(p)
    return out


def _same_seed_root(rec, dest, account):
    """Account B has two Drive mount names for the same shared run root."""
    if not isinstance(rec.get("source"), str) or not isinstance(rec.get("destination"), str):
        return False
    stored_source = Path(rec["source"])
    stored_dest = Path(rec["destination"])
    if (not stored_source.is_absolute() or ".." in stored_source.parts
            or stored_dest != stored_source / SUBDIR):
        return False
    actual_source = dest.parent.parent
    if stored_source == actual_source:
        return stored_dest == dest
    # START runs in account A using dsse_runs_B; B_T2 runs in account B
    # using dsse_runs. Only that exact sibling alias pair is accepted, for B.
    return (account == "B" and stored_source.parent == actual_source.parent
            and {stored_source.name, actual_source.name} == {"dsse_runs", "dsse_runs_B"})


def check(variant_runs, account):
    """Validate completed bootstrap identity without overwriting new run progress."""
    account = _account(account)
    dest = _path(variant_runs)
    if dest.name != VARIANT or dest.parent.name != "variants":
        raise ValueError(f"expected a {SUBDIR} run root: {dest}")
    marker = dest / MARKER
    _no_links(marker)
    if not dest.is_dir() or not marker.is_file():
        raise ValueError(f"variant bootstrap is incomplete: {marker} is missing")
    rec = json.loads(marker.read_text())
    if (not isinstance(rec, dict) or rec.get("schema") != 1 or rec.get("variant") != VARIANT or rec.get("h_policy") != VARIANT
            or rec.get("account") != account or not _same_seed_root(rec, dest, account)
            or not isinstance(rec.get("files"), dict)):
        raise ValueError(f"conflicting variant bootstrap identity: {marker}")
    # Seeded files can legitimately be replaced by the continuation. Its session
    # step records, not the historical seed hashes, validate those new outputs.
    return rec


def seed(source, account):
    """Atomically seed a separate run root, or keep a completed seed unchanged."""
    account = _account(account)
    source = _path(source)
    if not source.is_dir():
        raise ValueError(f"original run root is missing: {source}")
    if (source / MARKER).exists() or (source.name == VARIANT and source.parent.name == "variants"):
        raise ValueError("a variant run root cannot be used as the original source")
    parent = source / "variants"
    _no_links(parent)
    parent.mkdir(exist_ok=True)
    dest = parent / VARIANT
    _no_links(dest)
    lock = parent / f".{VARIANT}.seed.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"bootstrap already running (or stale lock; inspect before removing): {lock}") from None
    stage = None
    try:
        if dest.exists():
            check(dest, account)
            return str(dest)
        if account == "A" and not (source / "r1b_R_S0").is_dir():
            raise ValueError("account A requires its original r1b_R_S0 folder")
        prefix, markers = _prefix_plan(source) if account == "B" else ({}, {})
        trees = _trees(source)
        snapshots = {str(p): _inventory(p) for p in trees}
        source_top = {p.name for p in trees}
        stage = Path(tempfile.mkdtemp(prefix=f".{VARIANT}.seed-", dir=parent))
        records = {}
        for tree in trees:
            (stage / tree.name).mkdir()
            for rel in sorted(snapshots[str(tree)]):
                target_rel = f"{tree.name}/{rel}"
                _copy(tree / rel, stage / target_rel, records, target_rel)
        if account == "B":
            root = source / "r1b_K_S1"
            for rel, digest in {**prefix, **markers}.items():
                target_rel = f"r1b_K_S1/{rel}"
                _copy(root / rel, stage / target_rel, records, target_rel, expected=digest)
        # Stop on concurrent producers, including newly appearing independent
        # folders. No completed destination is exposed until these checks pass.
        if source_top != {p.name for p in _trees(source)}:
            raise ValueError("source folders changed while seeding; stop original notebooks and retry")
        for tree in trees:
            if snapshots[str(tree)] != _inventory(tree):
                raise ValueError(f"source folder changed while seeding: {tree}")
        if account == "B":
            later_prefix, later_markers = _prefix_plan(source)
            if prefix != later_prefix or markers != later_markers:
                raise ValueError("K_S1 prefix markers changed while seeding")
            for rel, digest in {**prefix, **markers}.items():
                p = source / "r1b_K_S1" / rel
                _no_links(p)
                if not p.is_file() or _sha(p) != digest:
                    raise ValueError(f"K_S1 prefix source changed while seeding: {p}")
        rec = dict(schema=1, variant=VARIANT, h_policy=VARIANT, label=LABEL, account=account,
                   source=str(source), destination=str(dest), seeded_utc=datetime.now(timezone.utc).isoformat(),
                   files=records, note="Original H+ folders retained. Stop original notebooks before bootstrap.")
        with open(stage / MARKER, "x") as f:
            json.dump(rec, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if dest.exists():
            raise ValueError(f"destination appeared during bootstrap: {dest}")
        os.rename(stage, dest)
        stage = None
        return str(dest)
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.rmdir()


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] not in ("seed", "check"):
        print("usage: r1b_htq_variant.py seed <original-runs> <A|B> | check <variant-runs> <A|B>", file=sys.stderr)
        return 2
    try:
        if args[0] == "seed":
            print(seed(args[1], args[2]))
        else:
            check(args[1], args[2])
            print(f"{VARIANT} bootstrap checked for account {args[2]}: {args[1]}")
    except (OSError, ValueError, KeyError) as e:
        print(f"H_tq variant bootstrap refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
