# Known limitations of 0.1.0b1

Status: release-candidate limitations after the recorded smoke validation and
component benchmark.

## Supported center

The documented beta supports continuous ordered parameters, fixed numeric
observation representations, one five-component diagonal-Gaussian Torch MDN
trainer, and portable NumPy CPU inference. This is deliberately narrower than
the broader class of simulator-based structural-inference problems.

## Known technical limits

- The MDN has diagonal covariance within each component and may not capture all
  dependence, tail, or boundary behavior.
- MAP is unavailable; only posterior mean and median point summaries are
  supported.
- Custom portable adapters must be importable and structurally serializable.
  The preferred load path supplies a compatible model; model-free
  reconstruction requires a trusted-artifact opt-in because it may import code
  selected by the artifact.
- Simulator source is not embedded or perfectly fingerprinted. Stateful
  callable instances require explicit identifiers and structured configuration,
  whose completeness remains the user's responsibility.
- Predictive checks after a fresh load require the compatible executable model.
- Accelerator training is optional and may be nondeterministic.
- Distribution-support checks are nearest-neighbor diagnostics rather than
  calibrated hypothesis tests.
- Large or variable-structure raw-data encoders, discrete/model-index inference
  through the new façade, distributed training, and multiple neural families
  are outside this beta.

## Evidence limits

The completed machine-readable smoke runs and timing benchmark are recorded in
`02_EXACT_POSTERIOR_VALIDATION.md` and `03_PERFORMANCE.md`. The larger manual
validation profiles have not been run. Historical research outputs are not
substituted for those current beta records.

The evidence is benchmark-specific. It cannot establish
general structural identification, correct real-world specification,
frequentist confidence coverage, robustness to arbitrary prior changes, or
universal speed superiority.

In particular, the SBC smoke record contains only 32 cases, with 10 or 11 in
each coarse support region, so its coverage and rank summaries are descriptive
and noisy. The structural exact-grid pass covers one predeclared synthetic
panel; it does not validate amortized accuracy across structural datasets or
models.

## Compatibility limits

Legacy alpha NPZ files remain trusted-local only and are not accepted as beta
artifacts. Expert fingerprint overrides are recorded and cannot bypass file
integrity, architecture, parameter-order, or representation-shape checks.
The custom-adapter trust opt-in is separate from fingerprint compatibility and
does not make unreviewed adapter code safe.

No large trained estimator is promised in ordinary Git history, and nothing in
this release-candidate workflow is published, tagged, or uploaded
automatically.

The Eight Schools smoke shows visible approximation error in the heterogeneity
parameter and joint dependence under the declared MDN. The canonical Rust
NFXP and dense-grid comparators validate data processing and dynamic-likelihood
conventions. In the completed full run, calibration and policy-functional
gates passed but the empirical parameter-mean, marginal-CDF, and joint-posterior
gates failed. It is therefore explicitly not a successful empirical neural
posterior replication. The Iskhakov-style three-panel smoke is also too small
for substantive bias, coverage, or amortization claims; its full campaign is
unrun.
