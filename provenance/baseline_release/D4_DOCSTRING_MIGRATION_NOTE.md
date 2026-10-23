# Off-manifold producer docstring migration

The released `d4_manifold.py` changes its module docstring and console-only reporting
messages so that they describe the same-grid, last-layer linear-span diagnostic actually
computed.  The numerical path, stored JSON/NPZ payload, masks, least-squares calls and every
reported scalar are unchanged.

- historical artifact-producer SHA-256:
  `4964dff75410cee27f6243bc6abbd18727c637c14967531a732489bcf9a642bb`
- corrected release-source SHA-256:
  `0b09b604ad69b3b6dc8a8a37239b1ae30637c5e406f52c55c98d4443d6119aa5`
- semantic AST SHA-256 for both after removing the module docstring and reporting-only
  `print(...)` calls:
  `85790e54f233473ebcf3bb7a1919edded3fe6b4f17b7b850123d1de69893b672`

The semantic digest uses the complete `ast.dump` representation, including empty optional
fields. Python 3.13 changed the display default to omit those fields, so the checker requests
`show_empty=True` when that keyword is available and uses the identical pre-3.13 default
otherwise. This is a serialization-compatibility rule only; it does not relax the exact release
source hash or remove any executable AST node from the comparison.

`analysis/offmanifold_summary.py --check` enforces the corrected source and semantic hashes,
records all three hashes in the generated summary, and checks every artifact's stored
`map_sha1` against the corresponding current admitted `params.pkl`. Any numerical edit or MAP
replacement therefore invalidates the historical artifacts instead of being silently presented
as their producer.
