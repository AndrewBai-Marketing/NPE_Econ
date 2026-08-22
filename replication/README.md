# Reproducing the 0.1.0b1 beta evidence

This directory reproduces the bounded evidence reported for the `structnpe`
0.1.0b1 public beta. It is a release-pipeline replication, not evidence that
an arbitrary simulator is identified, correctly specified, or accurately
approximated.

## What the runner covers

The smoke runner has separate, selectable stages for:

1. the complete package test suite;
2. wheel and source-distribution build;
3. wheel installation in a temporary environment;
4. source-distribution installation in a temporary environment;
5. the conjugate-normal exact-posterior comparison;
6. the 32-case SBC diagnostic emitted by that comparison;
7. the twelve-check structural exact-grid comparison;
8. a deliberately shifted synthetic observation that must trigger the
   training-support warning;
9. fixed-seed save/reload equality in a fresh Python process;
10. a small CPU neural-training and inference check; and
11. the frozen component benchmark.

The saved-estimator support check also verifies that the NumPy inference path
does not import Torch. Wheel and sdist installation inherit already prepared
dependencies from the replication environment and install only the target
artifact; dependency-resolution compatibility is covered by public CI.

## Environment

Use Python 3.11, 3.12, or 3.13 and install the development and CPU-neural
dependencies from the repository root:

```bash
python -m venv .venv-replication
. .venv-replication/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,replication]"
```

See [`environment/`](environment/) for dependency floors and the recorded
historical environment. Torch is required for the fitting stages. The saved
artifact, support warning, and loaded NumPy inference are CPU-only and do not
require Torch.

## Commands

Run every bounded smoke stage:

```bash
python replication/scripts/run_all.py --profile smoke
```

Run only the validation stages used by the manual CI workflow:

```bash
python replication/scripts/run_all.py \
  --profile smoke \
  --only conjugate_normal,sbc_32,structural_grid,support_warning \
  --report replication/results/ci_smoke_report.json
```

Stages can be selected or omitted without retraining unrelated estimators:

```bash
python replication/scripts/run_all.py --profile smoke --only package_tests
python replication/scripts/run_all.py --profile smoke --skip package_tests,benchmark
python replication/scripts/run_all.py --profile smoke --dry-run
```

`--use-recorded-evidence` permits comparison-only stages to read the committed
0.1.0b1 validation JSON when no fresh stage output exists. The report records
that provenance explicitly; it must not be described as a rerun.

The larger frozen validation budgets exist in the upstream threshold files,
but they have not been executed as release evidence. Consequently:

```bash
python replication/scripts/run_all.py --profile full
```

does no computation and writes `UNRUN_FUTURE_WORK`. This prevents an accidental
expensive launch or a fabricated full-profile claim. The fixed future budgets
are documented in [`configs/full_profile.json`](configs/full_profile.json).

## Runtime and outputs

On the historical 16-logical-CPU Linux machine, the substantive neural stages
took roughly 56 seconds for conjugate validation, 57 seconds for structural
validation, and 52 seconds for the component benchmark. Package tests took
about 90 seconds. Allow approximately 5–10 minutes for the complete smoke
profile after dependencies are installed; build and environment creation vary
by machine. Every subprocess has a fixed timeout in the frozen JSON configs.

Generated material is written below `replication/results/` and ignored by
Git, apart from the directory's `.gitignore`. The default report is:

```text
replication/results/smoke/report.json
```

It contains:

- each stage's status, command, duration, timeout, log, and artifacts;
- current and historical metrics, comparison rule, tolerance, and pass/fail;
- Python and dependency versions, CPU/platform metadata, and thread settings;
- the package source version and current Git commit/dirty state when Git is
  available.

The immutable historical values and comparison semantics are in
[`expected/release_0.1.0b1_metrics.json`](expected/release_0.1.0b1_metrics.json).
Timing is always descriptive and non-gating. For validation errors, the
reported tolerance is the original predeclared maximum from the corresponding
threshold file; the runner does not introduce a favorable threshold after a
rerun.

## Canonical validation extensions

These packages are separate from the eleven-stage release-pipeline runner.
They retain their own frozen configurations and compact, path-sanitized
evidence records.

| Package | Completed evidence | Status |
|---|---|---|
| [Eight Schools](eight_schools/README.md) | Exact quadrature/conditional-Gaussian comparator and bounded MDN smoke | Smoke passed its declared bounds; visible approximation error retained; larger profile unrun |
| [Rust (1987)](rust_1987/README.md) | Pinned public-data preprocessing, NFXP, dense grid, 20,000-simulation MDN, 200-case calibration, policy checks | Conventional and calibration/policy gates passed; empirical marginal and joint posterior gates failed; no successful empirical NPE claim |
| [Iskhakov-style Monte Carlo](iskhakov_2016/README.md) | Three panels at `beta=0.975` | Descriptive smoke only; six-beta by 250-panel campaign configured but unrun |

The Rust reference can be rebuilt after downloading the checksum-pinned public
group-4 file:

```bash
python -m replication.rust_1987.download_data
python -m replication.rust_1987.preprocess
python -m replication.rust_1987.nfxp \
  replication/rust_1987/data/processed/group4.csv
```

The completed full MDN evidence is summarized in
[`rust_1987/expected/smoke_metrics.json`](rust_1987/expected/smoke_metrics.json).
Its file name preserves the original evidence path; the record now includes
both the first smoke and the completed full campaign. Raw data, processed
rows, trained weights, and generated result directories are not distributed.

## Frozen smoke configurations

All runner configuration uses standard JSON; no YAML dependency is added.

- [`configs/conjugate_normal_smoke.json`](configs/conjugate_normal_smoke.json)
  mirrors the hash-pinned predeclared exact-validation smoke profile.
- [`configs/sbc_32.json`](configs/sbc_32.json) fixes the tiny SBC case count,
  draws, regions, bins, and seeds while retaining descriptive-only status.
- [`configs/structural_grid_smoke.json`](configs/structural_grid_smoke.json)
  mirrors the hash-pinned predeclared twelve-check structural profile.
- [`configs/cpu_benchmark.json`](configs/cpu_benchmark.json) fixes the
  historical timing configuration and declares it non-gating.
- [`configs/pipeline_smoke.json`](configs/pipeline_smoke.json) bounds package,
  build/install, warning, persistence, and small CPU-neural checks.

The runner refuses to start if either source threshold file no longer matches
its frozen SHA-256 or if a copied configuration or acceptance maximum drifts.

## Historical reference values

| Evidence | 0.1.0b1 reference |
|---|---:|
| Conjugate posterior-mean MAE | 0.01148 |
| Conjugate posterior-SD MAE | 0.00692 |
| Conjugate predictive-SD MAE | 0.00488 |
| Tiny-SBC bias / RMSE | -0.0200 / 0.2258 |
| Tiny-SBC 50/80/90/95% coverage | 0.4688 / 0.8125 / 0.8438 / 0.9688 |
| Structural joint-grid TV | 0.2521 |
| Exact / approximate policy mean | 0.23030 / 0.22928 |
| Structural predictive-cell maximum error | 0.00230 |
| Simulation time | 0.0439 s |
| Training excluding simulation | 52.19 s |
| One dataset, 10,000 draws, warm median | 0.570 ms |
| Batch of 32, warm median | 12.88 ms |
| Load time, warm-process median | 2.90 ms |

These timings describe one machine and are not acceptance thresholds. The
analytic conjugate comparator was faster per dataset, so the recorded
benchmark had no finite amortization break-even against that comparator.

## Scientific interpretation

| Result | Reproducible now | Scientific interpretation |
|---|---|---|
| Exact toy posterior | Yes | Basic posterior-approximation validation |
| 32-case SBC | Yes | Pipeline smoke test only |
| Structural grid | Yes | One fixed synthetic structural panel |
| CPU timing | Yes, machine-dependent | Deployment benchmark |
| General structural accuracy | No | Requires model-specific validation |
| Structural identification | No | Not supplied by the package |

The 32 SBC cases leave only 10 or 11 observations in each reported regional
slice. Coverage and rank summaries are therefore noisy and cannot support a
calibration claim. The structural-grid result checks one fixed synthetic panel:
policy and predictive functionals were close, while the joint-grid TV was
materially larger. It does not establish accuracy across observations or
transfer to another structural model.

The empirical Rust result supplies a stronger negative check: the conventional
likelihood and dense Bayesian reference were reproduced, and average
prior-predictive calibration and policy gates passed, while the empirical MDN
posterior still failed the frozen marginal and joint gates. A calibration pass
is therefore not treated as permission to suppress a dataset-specific
posterior failure.

## Nondeterminism and interpretation

Seeds fix simulation, training initialization, and posterior sampling in the
declared CPU runs. Exact floating-point results can still vary with Python,
NumPy, Torch, BLAS, compiler, and hardware versions. Neural weight files are
not compared bit for bit across environments. The authoritative gates are the
predeclared validation maxima; historical realized metrics are retained for
context. Save/reload equality is stricter: within one run and software
environment, fixed-seed posterior draws and the model fingerprint must match
exactly across the fresh process.
