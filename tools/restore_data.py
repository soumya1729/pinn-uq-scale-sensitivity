"""Restore the recorded saved arrays from the separate, checksum-verified data asset.

Run from anywhere: python tools/restore_data.py --archive /path/to/data.zip
This copies data only. It never trains a model or builds a Jacobian.
"""
from pathlib import Path, PurePosixPath
import argparse
import collections
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def destination(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or ":" in relative or "\\" in relative:
        raise ValueError(f"Unsafe data destination: {relative}")
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Data destination escapes repository: {relative}")
    return target


def main():
    manifest_path = ROOT / "data/manifest.json"
    public = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = public["entries"]
    groups = sorted({r["group"] for r in entries})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--group", nargs="+", choices=groups)
    parser.add_argument("--dest", type=Path, default=ROOT, help="Alternate extraction root for verification")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for group in groups:
            rows = [r for r in entries if r["group"] == group]
            print(f"{group}: {len(rows)} files, {sum(r['bytes'] for r in rows) / 2**20:.1f} MiB")
        return
    if args.archive is None:
        parser.error("--archive is required; the data asset is distributed separately")
    expected_asset_hash = (ROOT / "data/ASSET_SHA256.txt").read_text().split()[0]
    print("Checking data asset SHA-256...", flush=True)
    actual_asset_hash = sha256(args.archive)
    if actual_asset_hash != expected_asset_hash:
        raise ValueError("Data asset SHA-256 differs from this release")
    selected = [r for r in entries if args.group is None or r["group"] in args.group]
    args.dest.mkdir(parents=True, exist_ok=True)
    restored = skipped = 0
    verified_objects = {}
    with zipfile.ZipFile(args.archive) as z:
        if json.loads(z.read("manifest.json")) != public:
            raise ValueError("Archive and repository data manifests differ")
        for row in selected:
            target = destination(args.dest, row["path"])
            sha = row["sha256"]
            if target.is_file() and target.stat().st_size == row["bytes"] and sha256(target) == sha:
                skipped += 1
                verified_objects.setdefault(sha, target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if sha in verified_objects:
                shutil.copyfile(verified_objects[sha], target)
            else:
                with z.open(row["object"]) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            if target.stat().st_size != row["bytes"] or sha256(target) != sha:
                raise ValueError(f"Restored data identity differs: {row['path']}")
            verified_objects[sha] = target
            restored += 1
            if restored % 25 == 0:
                print(f"Restored {restored}; already verified {skipped}", flush=True)
    print(f"PASS: {len(selected)} data destinations verified ({restored} restored, {skipped} already present). No analysis was run.")


if __name__ == "__main__":
    main()
