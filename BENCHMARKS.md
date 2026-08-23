# Public-data benchmarks

This page compares posterior estimates with independent numerical references
on the two public examples maintained in this repository. A conventional
maximum-likelihood estimate is reported separately from Bayesian posterior
summaries.

## Results at a glance

### Rust bus replacement

| Method and estimand | Replacement cost | Maintenance slope |
| --- | ---: | ---: |
| Public NFXP maximum likelihood | 10.0749 | 2.2931 |
| This repository's NFXP maximum likelihood | 10.0749 | 2.2931 |
| Dense-grid posterior mean | 10.6095 | 2.5108 |
| Simulation-trained grid posterior mean (seed 1701) | 10.5832 | 2.4972 |

The public and independently reproduced NFXP rows agree. The two posterior
rows use the same model, 37-bus panel, and independent uniform priors. All five
fixed classifier seeds pass the frozen numerical posterior and policy
comparison limits. The classifier is a Rust-specific estimator, not the
generic `structnpe.fit` MDN.

### Eight Schools

| Parameter | Exact posterior mean | `structnpe.fit` mean | Difference |
| --- | ---: | ---: | ---: |
| Population mean `mu` | 6.5031 | 6.1148 | -0.3883 |
| Heterogeneity `tau` | 4.6855 | 6.0892 | +1.4037 |

The bounded Eight Schools smoke run passed its declared thresholds. It still
has visible approximation error, especially in heterogeneity and posterior
dependence, and is not presented as an exact match.

## Eight Schools model

The canonical data are

```text
y     = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18].
```

The maintained hierarchical model is

$$
\mu\sim\mathcal N(0,10^2),\qquad
\tau\sim\operatorname{HalfNormal}(10),
$$

$$
\theta_j\mid\mu,\tau\sim\mathcal N(\mu,\tau^2),\qquad
y_j\mid\theta_j\sim\mathcal N(\theta_j,\sigma_j^2).
$$

The reference integrates over \(\tau\) by dense one-dimensional quadrature
and uses the conditional Gaussian law for \(\mu\) and the school effects. The
five-component diagonal-Gaussian MDN is trained through the public API.

| Diagnostic | Result |
| --- | ---: |
| Standardized parameter-mean MAE | 0.187 |
| Standardized posterior-SD MAE | 0.147 |
| Maximum absolute correlation error | 0.366 |
| Absolute error in posterior mean of `tau` | 1.404 |

Run the checked bounded comparison from a repository checkout:

```bash
python -m pip install ".[replication]"
python replication/eight_schools/run_validation.py --profile smoke --quiet
```

The larger configuration remains unrun. Code and compact evidence are in
[`replication/eight_schools`](https://github.com/AndrewBai-Marketing/NPE_Econ/tree/main/replication/eight_schools).

## Rust bus-replacement model

The benchmark uses the public group-4 panel associated with Rust's engine
replacement model: 37 buses, 117 periods per bus, and 4,292 conditional choice
observations. The downloader is pinned to an immutable source commit and
SHA-256 checksum; raw bus histories are not redistributed.

The two estimated parameters are replacement cost \(RC\) and the
mileage-dependent maintenance slope \(c\). Transition probabilities and the
discount factor are fixed at the declared values, and

$$
RC\sim U(4,18),\qquad c\sim U(0.2,6).
$$

For observed exposure count \(n_s\) and replacement count \(r_s\) in mileage
state \(s\), the conditional choice posterior is proportional to

$$
\pi(\theta\mid r,n)
\propto \pi(\theta)
\prod_s p_s(\theta)^{r_s}
          [1-p_s(\theta)]^{n_s-r_s},
\qquad \theta=(RC,c),
$$

where \(p_s(\theta)\) is the dynamic replacement policy. The dense reference
evaluates this object on a \(141\times117\) trapezoidal grid. Its posterior
mean, standard deviation, and correlation are

$$
\mathbb E[\theta\mid r,n]=(10.6095,\,2.5108),
\qquad
\operatorname{sd}(\theta\mid r,n)=(1.4404,\,0.5862),
\qquad
\rho=0.9204.
$$

### Simulation-trained structured classifier

The positive comparison uses a \(29\times30\) parameter grid with trapezoidal
prior weights \(q_g\). At each class \(g\), it estimates the state policy from
\(M=500\) simulated fixed-exposure panels. The implementation draws the
aggregate counts

$$
K_{gs}\sim\operatorname{Binomial}(M n_s,p_s(\theta_g)),
\qquad
\widehat p_{gs}=\frac{K_{gs}+1/2}{M n_s+1},
$$

then computes

$$
\widehat w_g(r,n)
\propto q_g\prod_s
\widehat p_{gs}^{r_s}(1-\widehat p_{gs})^{n_s-r_s}.
$$

The aggregate draw is distributionally identical to summing \(M\) independent
panel-level Binomial counts. It avoids materializing 435,000 redundant panel
rows; it does not insert the exact policy into the classifier likelihood.

| Seed | Posterior mean `RC` | Posterior mean slope | Max marginal CDF error | Joint TV |
| ---: | ---: | ---: | ---: | ---: |
| 1701 | 10.5832 | 2.4972 | 0.08645 | 0.13401 |
| 1702 | 10.6331 | 2.5224 | 0.07328 | 0.10996 |
| 1703 | 10.6408 | 2.5199 | 0.07688 | 0.11974 |
| 1704 | 10.5658 | 2.4803 | 0.08374 | 0.14653 |
| 1705 | 10.6245 | 2.5285 | 0.06959 | 0.06073 |

Across the five seeds, the worst mean error was 0.0521 reference standard
deviations, the worst policy-mean error was 0.00122, and the worst policy-band
endpoint error was 0.00203. The frozen limits are 0.25, 0.03, and 0.06,
respectively. Every grid class lies inside the declared prior support.

This is a deliberately model-specific, likelihood-aligned finite-grid
classifier. It is not an amortized estimator for arbitrary panels and is not
the generic `structnpe.fit` neural density estimator. It was introduced after
diagnosing the MDN failure, so it is a transparent reconstruction rather than
a prospective estimator-selection result; the pre-existing numerical limits
were not changed.

### Preserved generic-MDN diagnostic

The earlier 20,000-simulation generic diagonal-MDN result remains checked in:

| Diagnostic | Generic MDN | Frozen limit | Status |
| --- | ---: | ---: | --- |
| Maximum marginal CDF discrepancy | 0.473 | 0.100 | fail |
| Coarsened joint total variation | 0.514 | 0.150 | fail |
| Maximum mean error in reference SDs | 1.565 | 0.250 | fail |
| Policy-mean maximum absolute error | 0.0172 | 0.0300 | pass |

Its posterior mean was \((12.8636,3.2528)\). That error lies along the
reference posterior's strong positive ridge. Average prior-predictive
calibration passed, but it did not guarantee accuracy at this empirical panel.
This failure is estimator evidence; it is not evidence that the public NFXP
replication, dense likelihood, or data processing is wrong.

### Reproduction

```bash
python -m replication.rust_1987.download_data
python -m replication.rust_1987.preprocess
python -m replication.rust_1987.nfxp \
  replication/rust_1987/data/processed/group4.csv
python -m replication.rust_1987.structured_classifier \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/structured_classifier_metrics.json
```

The dense-grid, generic train/infer, comparison, and calibration commands are
documented in
[`replication/rust_1987`](https://github.com/AndrewBai-Marketing/NPE_Econ/tree/main/replication/rust_1987).
Compact machine-readable results are
[`expected/structured_classifier_metrics.json`](https://github.com/AndrewBai-Marketing/NPE_Econ/blob/main/replication/rust_1987/expected/structured_classifier_metrics.json)
and
[`expected/smoke_metrics.json`](https://github.com/AndrewBai-Marketing/NPE_Econ/blob/main/replication/rust_1987/expected/smoke_metrics.json).

## Interpretation

The checked evidence shows that the public Rust maximum-likelihood result is
reproduced, that the structured simulation classifier closely matches the
declared Bayesian reference across five seeds, and that the current generic
diagonal MDN is not reliable for that same empirical posterior. Eight Schools
provides a separate, direct public-API smoke test.

None of these results establishes structural identification, correct model
specification, causal interpretation, or universal posterior accuracy. A new
application needs its own comparator, simulation-based calibration, or other
predeclared validation study.
