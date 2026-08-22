# Release readiness for `structnpe` 0.1.0b1

Status: **`GITHUB_BETA_BLOCKED` only on final Git identity and commit-bound
release provenance**  
Assessment date: 2026-08-22

## Outcome

The standalone source-only public tree has passed the bounded software,
example, replication, security, distribution, and documentation gates required
before its first public commit. The estimator-development cycle remains frozen.
No tag, remote, GitHub release, PyPI upload, or other publication has occurred.

The remaining blocker is mechanical but real: this machine has no configured
Git author email. A public or GitHub `noreply` address must be supplied by the
author before the first commit can be created. Until that commit exists, the
release manifest cannot truthfully record a source commit, and artifacts built
from the eventual commit cannot yet be called authoritative release assets.

## Public boundary and metadata

- Public root: the standalone `structnpe/` directory.
- Package version: `0.1.0b1`; supported Python: 3.11--3.13.
- License: MIT, with the pre-existing copyright attribution to Andrew Bai.
- `pyproject.toml`, `LICENSE`, `CITATION.cff`, README badge/text, and package
  metadata agree on MIT and version `0.1.0b1`.
- `CITATION.cff` has structurally valid 1.2.0 software metadata and intentionally
  contains no invented DOI or email address.
- The public export contains no other research namespace, private dataset,
  inherited handoff history, or generated empirical row data. The complete
  boundary audit is in
  [`07_GITHUB_PUBLICATION_AUDIT.md`](07_GITHUB_PUBLICATION_AUDIT.md).

## Current verification record

- Complete source suite: **101 passed, 3 expected skips**, with one documented
  legacy deprecation warning. The skips were the two pre-build distribution
  checks and the opt-in pinned-network regression.
- Focused post-fix security suite: **59 passed, 1 expected network skip** under
  the independent audit.
- Eleven-stage replication smoke: **11/11 stages passed**, **6/6 gating metric
  comparisons passed**, and no gating comparison failed.
- Generated smoke report SHA-256:
  `8810a6cf63acf8162fd66bad86a633e8b658cd20281ba528a84069fd23fa19ae`.
- README snippet check and executable-example inventory passed.
- The built wheel and normalized sdist passed the distribution-boundary tests;
  both installed and imported successfully in the smoke runner.

The checked-in, path-sanitized beta records retain these hashes:

- conjugate-normal metrics:
  `0200b62aa36e065e74ec8bf676f596ac8bd603f2d7e2cdb652f95bd210156f95`;
- structural-grid metrics:
  `9a22c97e3ad6ae4b516bf0b010e473006d286d6ffc4840c6c8073c7f46659704`;
- component benchmark:
  `14236ee3ba2af2f5cbd6f173555aee82e2bcc5ab10084e6ffe364859ec8daaa6`.

Timing in a new smoke run remains descriptive rather than gating. The release
workflow will rebuild and test the exact tag on Python 3.11, 3.12, and 3.13,
normalize and boundary-test the exact upload sdist, clean-install the exact
wheel and sdist, verify checksums and manifest provenance, and create only a
manually authorized GitHub prerelease. There is no PyPI publication path.

## Canonical replication status

- The independent Rust data transformation reproduces 4,329 panel rows and the
  declared historical transition counts. NFXP returns replacement cost
  `10.07494221` and maintenance slope `2.29309298`; the completed dense grid
  supplies the same-prior Bayesian reference.
- The frozen full Rust MDN passes 200-case prior-predictive calibration and
  policy-functional gates but fails the empirical parameter-mean,
  marginal-CDF, and joint-distribution gates. It is explicitly not a successful
  empirical NPE posterior replication.
- Eight Schools is a bounded comparator smoke with visible approximation
  error.
- The Iskhakov-style result is a three-panel descriptive smoke; its complete
  six-discount-factor by 250-panel campaign is configured but unrun. MPEC is
  unsupported.

## Security and archive hardening

The beta estimator loader now rejects architecture declarations that would
exceed the existing array, member-count, or decompression limits before it
enumerates weight shapes. The sdist normalizer rejects symlink inputs,
nonportable or traversal-like member names, wrong roots, duplicate canonical
names, and special members. The pinned Rust downloader validates cached files
without following symlinks or reading an unbounded payload.

GitHub Actions are referenced by immutable commit SHA. The release publisher
re-resolves and peels the remote tag immediately before publication, compares
it with the build job's commit, verifies `SHA256SUMS` and every manifest entry,
and requires a separate boolean authorization. Build backend versions are
fixed for the release job.

## Conditions for changing the status

After the author supplies an intentional public Git email:

1. create the first source-only commit from the reviewed index;
2. check out that exact commit into a clean directory;
3. rebuild and normalize the wheel and sdist;
4. rerun clean wheel/sdist and installed-example validation;
5. generate `release_manifest.json` and `SHA256SUMS` with that exact commit;
6. review the bundle and, only under separate authorization, create the
   protected `v0.1.0b1` tag and GitHub prerelease.

Until those steps are complete, the correct status remains
`GITHUB_BETA_BLOCKED`.
