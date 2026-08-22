# Limitations

`structnpe` performs approximate Bayesian inference under a maintained modeling
system. Its conclusions are conditional on the supplied prior, simulator,
parameter definitions, and observation representation.

- It does not prove structural identification or distinguish observationally
  equivalent mechanisms by naming them differently.
- It does not establish that the simulator is correctly specified.
- Calibration under the training distribution does not imply reliability for
  an out-of-support observation.
- A support warning is diagnostic, not a formal hypothesis test.
- Training cost may dominate for a one-off dataset. Amortization matters only
  when the estimator is reused enough times.
- The primary beta posterior is a five-component diagonal-Gaussian MDN. It can
  still miss posterior geometry or tails.
- Summaries or embeddings can discard information; the target is conditional
  on the stored representation.
- Posterior intervals are Bayesian credible intervals, not confidence
  intervals. `posterior_sd` is not a standard error.
- MAP, frequentist confidence sets, automatic model choice, and automatic
  simulator introspection are not implemented.
- Stateful prior and simulator objects require user-declared versioned
  identifiers and structured configuration; the package does not infer whether
  those declarations fully describe executable behavior.
- A pretrained estimator is tied to its fingerprint, ordered parameters, and
  observation adapter.
- CPU inference is the supported portable path. Accelerator determinism is not
  guaranteed.
- Portable custom adapters require importable code and structured state. Safe
  loading prefers a caller-supplied compatible model; model-free reconstruction
  requires an explicit trust opt-in because it may import artifact-selected
  code.
- In the completed full Rust experiment, prior-predictive calibration and
  policy gates passed while the empirical posterior failed frozen marginal and
  joint agreement gates. Passing calibration does not guarantee accuracy for a
  particular empirical observation.
- The Iskhakov-style Monte Carlo result uses only three panels at one discount
  factor. Its full six-factor by 250-panel campaign remains unrun.

Legacy finite-grid, ridge-Gaussian, model-index, project-file, and CLI paths are
retained for compatibility but are not the documented primary beta workflow.
Passing software tests or selected predictive checks is not a scientific
validity claim.
