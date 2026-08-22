# structnpe

![Status: public beta](https://img.shields.io/badge/status-public%20beta-orange)
![Python: 3.11--3.13](https://img.shields.io/badge/python-3.11--3.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

`structnpe` provides amortized Bayesian inference for simulator-based
structural models. Train an estimator on simulated datasets, then obtain named
joint posterior draws and conventional Bayesian summaries for new datasets at
neural-network inference speed.

**Train on simulations once. Return a full approximate posterior for each new
dataset in milliseconds under the documented benchmark conditions.**

The package supports bring-your-own simulators, CPU-only deployment from saved
estimators, posterior summaries and transformations, batch inference, and
posterior predictive checks.

> **Beta status**
>
> `structnpe` 0.1.0b1 is suitable for pilot use. Inference is conditional on
> the supplied simulator, prior, observation representation, and trained
> estimator. Every serious application requires model-specific posterior
> validation.

## What you get

For each observed dataset, `structnpe` returns named, aligned draws from the
trained approximate joint posterior. Those draws support:

- posterior means, medians, standard deviations, and credible intervals;
- covariance, correlation, and posterior probabilities;
- arbitrary transformed parameters and decision-relevant functionals;
- elasticities, willingness-to-pay measures, profits, welfare, and policy
  counterfactuals when supplied by the model;
- posterior predictive distributions.

The parameter table is a convenience summary of the draws, not a replacement
for them. Posterior standard deviations are not standard errors, and credible
intervals are not confidence intervals.

## Installation

Install the base package for NumPy CPU inference from a saved estimator:

```bash
python -m pip install .
```

Install the neural extra when training an estimator:

```bash
python -m pip install ".[neural]"
```

Python 3.11, 3.12, and 3.13 are supported. See the
[installation guide](docs/installation.md) for clean-environment checks.

## 60-second quickstart

The repository includes a roughly 60 KB, pickle-free estimator trained only
on synthetic data. Loading and inference use NumPy on CPU and do not import
Torch.

```python
from pathlib import Path

import numpy as np

from structnpe import load_estimator

observed_data = np.array(
    [32, 1, 27, 3, 18, 7, 12, 4, 6, 10],
    dtype=float,
)
estimator = load_estimator(Path("examples/assets/demo_estimator"))
draws_requested = 10_000
seed = 123

result = estimator.infer(
    observed_data,
    draws=draws_requested,
    seed=seed,
    device="cpu",
)

columns = [
    "parameter",
    "posterior_mean",
    "posterior_sd",
    "median",
    "q025",
    "q975",
]
print(result.summary()[columns].to_string(index=False))
draws = result.to_dataframe()
```

The checked-in script generated this fixed-seed output:

```text
       parameter  posterior_mean  posterior_sd   median     q025    q975
maintenance_cost        0.552179      0.103336 0.539629 0.342964 0.70229
replacement_cost        4.689264      0.724780 4.742391 3.037325 5.73117
```

`draws` has 10,000 rows with a draw index and the two aligned parameter
columns. Run the complete deterministic example with:

```bash
python examples/01_quickstart_saved_estimator.py
```

## Use the full posterior

The [Bayesian workflow example](examples/02_bayesian_workflow.py) calculates
ordinary posterior quantities directly from the aligned draws:

```python
import pandas as pd

summary = result.summary()
draws = result.to_dataframe()
names = list(estimator.parameter_names)
covariance = pd.DataFrame(result.covariance(), index=names, columns=names)
correlation = pd.DataFrame(result.correlation(), index=names, columns=names)

probability = float((draws["maintenance_cost"] > 0.4).mean())
cost_ratio = draws["replacement_cost"] / draws["maintenance_cost"]
ratio_interval = cost_ratio.quantile([0.025, 0.5, 0.975])
```

When the compatible executable model is attached, the same result can drive a
posterior predictive check:

```python
predictive = result.predictive_check(replications=200, seed=456)
```

These are independent mixture draws, not MCMC chains; the package therefore
does not report invented R-hat or autocorrelation diagnostics.

## Bring your own simulator

A model declares an ordered parameter vector, prior, simulator, and fixed
observation representation. Simulators may accept one parameter row or an
entire training batch.

```python
import numpy as np

from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit

N_OBSERVATIONS = 30


def prior(n, rng):
    return np.column_stack([
        rng.uniform(-2.95, 2.95, n),
        rng.uniform(0.15, 1.95, n),
    ])


def simulator(theta, rng):
    theta = np.asarray(theta)
    if theta.ndim == 1:
        return rng.normal(theta[0], theta[1], size=N_OBSERVATIONS)
    return rng.normal(
        theta[:, [0]], theta[:, [1]], size=(len(theta), N_OBSERVATIONS)
    )


model = StructuralModel(
    prior=prior,
    simulator=simulator,
    parameters=[
        ParameterSpec("beta", lower=-3.0, upper=3.0),
        ParameterSpec("sigma", lower=0.1, upper=2.0, unit="outcome units"),
    ],
    observation_adapter=SummaryAdapter(
        statistics=("mean", "std"),
        expected_shape=(N_OBSERVATIONS,),
    ),
    prior_id="structnpe.examples.normal_location_scale.prior.v1",
    simulator_id="structnpe.examples.normal_location_scale.simulator.v1",
    prior_config={"beta": [-2.95, 2.95], "sigma": [0.15, 1.95]},
    simulator_config={"observations": N_OBSERVATIONS, "likelihood": "normal"},
    batched_simulator=True,
)
estimator = fit(model, simulations=2_000, epochs=20, seed=303, device="cpu")
estimator.save("saved/normal_estimator")
```

The complete [custom-simulator example](examples/03_custom_simulator.py)
trains, infers, saves, reloads, and checks exact seeded draw equality on CPU.
Its defaults are a bounded demonstration configuration, not a validation
claim.

## Train once, reuse repeatedly

Training is the amortized up-front cost. Once trained, the estimator can be
reused for many datasets generated under the same model and prior:

```text
one trained estimator
  -> many markets, waves, simulated datasets, or experimental cells
  -> separate named joint posterior draws for every dataset
```

See [batch inference](examples/04_batch_inference.py) for the actual batched
API, deterministic seeded draws, and elapsed inference time measured separately
from loading and training.

In the release CPU component benchmark, a loaded estimator generated 10,000
draws for one dataset in a median 0.57 ms after training. Training took about
52 seconds in that small synthetic demonstration; a warm batch of 32 took a
median 12.88 ms and loading took a median 2.90 ms. Real structural simulators
can cost much more. There was no finite break-even against the benchmark's
closed-form analytic comparator, and the package claims no universal speed
advantage for one-off estimation. Full conditions are in the
[performance record](docs/release/03_PERFORMANCE.md).

## Validation

`structnpe` returns a complete approximate posterior over the trained parameter
vector. Accuracy is model- and training-specific. The 0.1.0b1 evidence includes:

- an exact conjugate-normal comparison;
- a 32-case simulation-based-calibration pipeline smoke test;
- an exact-grid structural smoke comparison;
- posterior predictive checks;
- a training-support warning demonstration.

The structural smoke passed all 12 predeclared checks. Its policy mean and
predictive cells were close to the exact comparator, but the declared
nearest-grid joint total variation was materially larger at **0.2521**. That
result is visible because application-specific joint-posterior validation is
essential.

These checks validate named release workflows, not every future simulator.
Read the [exact-posterior record](docs/release/02_EXACT_POSTERIOR_VALIDATION.md),
[validation guide](docs/validation.md), and [limitations](docs/limitations.md),
or run [the structural example](examples/05_structural_grid_validation.py).

## Replication

The frozen beta pipeline has one developer-machine entry point:

```bash
python -m pip install ".[replication,dev]"
python replication/scripts/run_all.py --profile smoke
```

It reruns the package and distribution gates, analytic and structural
comparators, tiny SBC, support warning, save/reload check, CPU neural path, and
component benchmark, then writes a machine-readable comparison report under
`replication/results/`. Timing references are descriptive across machines.

The repository also contains an
[Eight Schools validation](replication/eight_schools/README.md) against
one-dimensional quadrature plus conditional Gaussian calculations. The smoke
run deliberately reports the visible approximation errors of the declared
five-component diagonal-Gaussian MDN; it is not presented as an exact recovery
result. The larger profile is frozen but unrun.

### Canonical Rust evidence: comparator pass, empirical NPE failure

The [Rust (1987) bus-replacement package](replication/rust_1987/README.md)
pins the public group-4 data transformation and independently reproduces the
canonical NFXP estimates: replacement cost `10.07494` and maintenance slope
`2.29309`. A completed `141 x 117` dense grid supplies the Bayesian reference
under the declared prior.

The full 20,000-simulation MDN run is intentionally reported as a **failed
empirical posterior replication**:

| Check | Dense-grid reference / limit | `structnpe` result | Gate |
|---|---:|---:|---|
| Posterior mean, replacement cost | 10.6095 | 12.8636 | fail |
| Posterior mean, maintenance slope | 2.5108 | 3.2528 | fail |
| Maximum marginal CDF discrepancy | at most 0.100 | 0.473 | fail |
| Coarsened joint total variation | at most 0.150 | 0.514 | fail |
| Policy-mean maximum error | at most 0.030 | 0.0172 | pass |

The separate 200-case prior-predictive calibration gates passed, as did the
policy-functional gates, but every promotion gate was required. The package
therefore makes no successful empirical Rust NPE claim. This is a useful
warning that average calibration and policy agreement need not imply an
accurate posterior at one empirical panel.

An [Iskhakov-style Monte Carlo](replication/iskhakov_2016/README.md) has only
been run as a three-panel, one-discount-factor smoke. Its bias, coverage, and
break-even numbers are descriptive; the frozen six-factor, 250-panel campaign
is configured but unrun. MPEC is not implemented in either replication.

See the [replication guide](replication/README.md) for frozen configurations,
expected runtimes, outputs, nondeterminism, and the distinction between smoke
tests and scientific validation.

## Limitations

- Inference is conditional on the supplied simulator, prior, representation,
  and estimator; the package does not prove identification or specification.
- The beta posterior family is a five-component diagonal-Gaussian MDN. It may
  miss dependence, tails, boundaries, or multimodal geometry.
- A support warning is a nearest-neighbor diagnostic, not a formal test.
- The 32-case SBC is a pipeline smoke test, not a calibration campaign.
- The structural-grid evidence covers one fixed synthetic panel.
- The full empirical Rust MDN posterior failed its frozen parameter-mean,
  marginal-distribution, and joint-distribution gates, despite passing its
  prior-predictive calibration and policy-functional gates.
- The Iskhakov-style result has only three panels at one discount factor; the
  complete campaign has not been run.
- Training can dominate a one-off analysis, and simulator costs are
  application-specific.
- Saved estimators are tied to their model fingerprint and observation
  representation.
- MAP, automatic model choice, and frequentist confidence sets are not
  implemented.

The full boundary is documented in [known limitations](docs/limitations.md).

## Citation

The project is MIT-licensed. If it supports published work, cite the software
metadata in [CITATION.cff](CITATION.cff):

```text
Andrew Bai. (2026). structnpe: A simulator-to-posterior toolkit for
structural models (Version 0.1.0b1) [Computer software].
```

No DOI or hosted release is claimed for this local prerelease candidate.
