"""Check the independent study's frozen input copies; standard library only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    tags = [r["tag"] for r in manifest["runs"]]
    assert len(tags) == len(set(tags)) == 17
    assert sorted(t for group in manifest["groups"] for t in group) == sorted(tags)
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if ROOT not in path.resolve().parents or path.is_symlink():
            raise ValueError(f"Input is outside this independent study: {path}")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], str(path)
    print(f"Verified {len(manifest['files'])} independent input copies for 17 checkpoints.")


if __name__ == "__main__":
    main()
