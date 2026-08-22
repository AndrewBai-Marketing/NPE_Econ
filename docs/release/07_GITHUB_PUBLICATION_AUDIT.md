# GitHub publication audit for `structnpe` 0.1.0b1

Audit date: 2026-08-22  
Audited root: standalone source-only `structnpe/` export  
Decision: public file boundary accepted; first commit blocked on intentional
Git author email

## 1. Source boundary

The public tree was assembled by selecting package-specific files from the
June cloud handoff and adding public examples, replication infrastructure,
tests, documentation, and GitHub configuration. The whole handoff was not
copied, and no history was imported merely to create commit count.

The following source paths or path classes were excluded in full:

- handoff administration: `CLOUD_FILELIST.txt`, `CLOUD_HANDOFF.md`, and
  `CLOUD_MANIFEST.txt`;
- build and cache state: `.lake/`, `.pytest_cache/`, `dist/`, `build/`, all
  egg-info, `__pycache__/`, bytecode, notebook checkpoints, and local virtual
  environments;
- formal/research projects: `NpeCompilerLean*`, `research/`, `artifacts/`,
  `paper/`, `paper_artifacts/`, and `paper_latex/`;
- data and empirical planning: `data_manifest/` and all non-public handoff
  datasets, outcome files, reference packets, execution packets, and scheduler
  logs;
- unrelated packages: `src/airline_compiler/`, `src/ddc_npe/`, and
  `src/marketing_compiler/`;
- unrelated research tests, scripts, configurations, and the handoff's broad
  research-document archive;
- obsolete handoff examples outside the curated installed-wheel examples and
  the retained `tiny_ddc` compatibility project;
- trained validation estimators and cached checkpoints under
  `validation/*/output/estimator/`;
- raw and processed Rust rows, Rust/Iskhakov generated results, Eight Schools
  generated figures/weights, and all general replication-run outputs;
- ordinary local release products: root `dist/`, `release-assets/`,
  `release_manifest.json`, and `SHA256SUMS`.

The separate later `NPE/` Git working tree was excluded in full. It contains a
different, dirty research history and has no configured remote. Its research
and mechanism work was neither copied nor modified.

## 2. History search and decision

The June handoff has no `.git` directory. A read-only search covered every
other Git worktree under the project directory, including reachable historical
paths; none contained `structnpe` or its package metadata. The only other
filesystem copy was the non-Git June handoff.

A new repository was therefore initialized on branch `main` inside the
standalone public root. It currently has no commit. This empty history is safe
by construction, but publication provenance is incomplete until an intentional
author email is supplied and the reviewed index is committed. No remote is
configured.

## 3. Private-reference corrections

The public snapshot contains no private username, home-directory reference, or
private email. Exact corrections made while assembling the public files were:

- `validation/exact_example/output/validation_metrics.json`: the recorded
  `best_checkpoint` was changed from a machine-absolute handoff path to
  `validation/exact_example/output/estimator/best_estimator`;
- `validation/structural_example/output/validation_metrics.json`: the same
  field was changed to its repository-relative path;
- generated replication JSON now records the Python executable basename rather
  than `sys.executable`, keeps stage paths repository-relative, and omits
  installer stdout that can contain temporary paths;
- Eight Schools, Rust, and Iskhakov committed evidence was reduced to compact,
  repository-relative records; generated manifests, row data, model weights,
  figures, and local result JSON were excluded;
- the synthetic demonstration estimator has `best_checkpoint: null` and only
  documented JSON/NPZ members.

Strings such as `private-user` in archive tests are deliberate synthetic
sentinels proving that ownership is removed; they are not identity metadata.

## 4. Authorship, license, and provenance

The following author field is intentionally retained:

- **Andrew Bai** in `LICENSE`, `CITATION.cff`, `pyproject.toml`, and the README
  citation. The name and MIT copyright statement already existed in the source
  handoff and were not inferred from the machine username.

No email and no DOI are published. `CITATION.cff` validates as CFF 1.2.0
software metadata. The root license, project metadata, citation metadata, and
README all say MIT.

The Rust replication code is independently written. It does not redistribute
the public raw group-4 file or processed rows; the downloader pins an immutable
OpenSourceEconomics commit, URL, byte count, and SHA-256. The documentation
identifies the preprocessing convention and comparator. Other validation data
is synthetic or the standard eight-number Eight Schools example.

## 5. Secret, credential, binary, and path scan

None of `gitleaks`, `trufflehog`, `detect-secrets`, or `git-secrets` was
installed, so no claim is made that one of those tools ran. Local read-only
scans covered:

- common access-token, private-key, cloud-credential, password-assignment, and
  `.env` patterns;
- home-directory paths, the machine username, handoff names, and private email
  patterns;
- symlinks, FIFOs, sockets, devices, files over 2 MiB, pickle/joblib objects,
  Torch checkpoints, and notebook outputs;
- archive member names, types, ownership, permissions, and package roots.

No credential, key, `.env`, special file, file over 2 MiB, private path, or
undocumented executable payload was found in the proposed index. The tracked
demonstration estimator is about 60 KiB, synthetic, checksum-verified, and
pickle-free.

## 6. Largest proposed tracked files

The largest files in the reviewed public tree are small source or declared
evidence files:

| Bytes | Path |
|---:|---|
| 59,829 | `src/structnpe/estimator.py` |
| 33,496 | `validation/structural_example/output/validation_metrics.json` |
| 33,443 | `validation/exact_example/output/validation_metrics.json` |
| 26,654 | `examples/assets/demo_estimator/state.npz` |
| 26,591 | `replication/scripts/run_all.py` |
| 25,661 | `src/structnpe/schema.py` |
| 22,782 | `src/structnpe/_artifacts.py` |
| 22,464 | `validation/structural_example/run_validation.py` |
| 17,101 | `src/structnpe/results.py` |
| 16,677 | `examples/assets/demo_estimator/manifest.json` |
| 16,611 | `examples/assets/demo_estimator/weights.npz` |
| 13,983 | `replication/rust_1987/preprocess.py` |

No proposed tracked file exceeds 60 KiB.

## 7. Archive and Git-mode audit

The workspace filesystem maps ordinary source files to an owner-executable
mode. That host-level mode is not appropriate public metadata. Every proposed
Git blob is therefore forced to mode `100644` in the index. The source archive
is independently rewritten with directories `0755`, files `0644`, UID/GID
zero, blank owner/group names, a fixed timestamp, one exact root, and only
regular files/directories.

The normalizer rejects symlink or special-file inputs, traversal and
nonportable Windows/UNC member names, canonical duplicates, unexpected roots,
and bounded-size violations. Distribution tests independently enforce the
public namespace and reject research, data, generated results, caches, and
executable payload types.

## 8. Public-content conclusion

The file boundary is suitable for a first public beta commit. Remaining
scientific limitations are disclosed, including the structural-grid joint TV,
the failed empirical Rust posterior gates, the small SBC, the bounded Eight
Schools and Iskhakov smokes, the diagonal-Gaussian mixture family, and the lack
of identification/specification guarantees.

Publication is not yet authorized or provenance-complete. The first commit,
tag, manifest, and authoritative artifact hashes remain blocked on the author's
intentional public Git email and subsequent commit-bound rebuild. Nothing has
been pushed, tagged, released, or published.
