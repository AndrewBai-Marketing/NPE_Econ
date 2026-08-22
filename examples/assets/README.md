# Demonstration estimator

`demo_estimator/` is a small, safe-format estimator used by the public
quickstart. It was trained solely on simulations from the five-state
replacement model in `validation/structural_example/run_validation.py`; it
contains no empirical or private data.

The artifact is a checksummed directory, not a pickle or executable archive.
It contains exactly JSON metadata, NumPy numeric arrays, and the completion
marker required by the public artifact parser. Its model fingerprint is
`c1371104ce7303d2a37688c3a9a98987f5f8f6cf059a897fe9dd680b2663e75d`
and its estimator fingerprint is
`6a1ce8a78a46428b83c776b26cc4c0d606f19d84b962fd3b9a0dfe4eed7cb99b`.

| File | SHA-256 |
| --- | --- |
| `COMPLETE` | `b4870a1d744875fa11f8edebaf85f76bff47a9505a61b3761b5b2c09c71b70e5` |
| `manifest.json` | `9c231da1bed5c8b5747a4f3da60f49cb1a194126ee7027202866300231970171` |
| `state.npz` | `3886b554574694ad1dfd5748ec3083cdabb50a349cefa4592da214e6598a8482` |
| `weights.npz` | `ff840237f7a25a6b00a88be345acc87a3173bbc4ac1b0280216440ab8e949446` |

The four artifact files occupy about 60 KB. The full validation record and
the limitations of this smoke estimator are documented in
`docs/release/02_EXACT_POSTERIOR_VALIDATION.md`.

