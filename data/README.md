# Saved data for numerical replay

The code repository and its compact CSV/JSON reference records are separate from
the array/checkpoint asset. The asset is:

`pinn_uq_saved_data_20261022.zip`

Its SHA-256 is in `ASSET_SHA256.txt`; `manifest.json` gives every destination,
purpose group, byte size and content hash. The ZIP stores each unique object
once and maps it to all required historical consumer paths. No machine-local
source paths or download credentials are needed to restore it.

After obtaining the asset, run from the repository root:

```bash
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip
```

To list groups or restore only selected groups:

```bash
python tools/restore_data.py --list
python tools/restore_data.py --archive /path/to/pinn_uq_saved_data_20261022.zip --group "saved outputs: round2" "theta-centred: production fields and shifts" "theta-centred: frozen inputs"
```

The command verifies the entire asset checksum, then verifies each restored
destination. It copies data only: it never trains, builds a Jacobian or starts
a cluster job. The full restored layout occupies about 2.38 GB because multiple
historical readers expect independent paths. The unique object content is about
1.65 GB. Large restored arrays are ignored by Git.

The separate 276 GB retained native factors/grid archive is **not** part of this
asset and is not needed for ordinary numerical replay of the paper. Recorded
retention manifests document that separate archive.

Historical scientific records may contain the original machines' source paths.
Those paths are preserved as hash-bound execution provenance, not as public
download locations or dependencies. Use this portable manifest and restoration
command; the old local-only copying utility is not distributed.

Download the ZIP from this repository's **Releases → v1.0.0 → Assets**. The
matching `ASSET_SHA256.txt` is also supplied there. Source code stays in Git;
the larger data ZIP is a release asset.
