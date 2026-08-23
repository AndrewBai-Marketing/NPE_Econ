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

| Result | Population mean `mu` | Heterogeneity `tau` |
| --- | ---: | ---: |
| Deterministic quadrature posterior mean | 6.4720 | 4.7531 |
| Mean of five `structnpe.fit` posterior means | 6.4178 | 4.7042 |
| Range across the five fitted posterior means | [6.0483, 6.9624] | [4.5063, 5.0521] |

The mean and range in the last two rows summarize stochastic training runs;
they are not an ensemble posterior or a posterior uncertainty interval.
All five fixed seeds pass the committed numerical accuracy gates. This is a
comparison of the marginalized two-parameter hyperposterior, not a claim that
the current diagonal mixture can reliably learn every centered hierarchical
parameterization.

## Eight Schools model

The canonical data are

```text
y     = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18].
```

The hierarchical model is

$$
\mu\sim\mathcal N(0,10^2),\qquad
\tau\sim\operatorname{HalfNormal}(10),
$$

$$
\theta_j\mid\mu,\tau\sim\mathcal N(\mu,\tau^2),\qquad
y_j\mid\theta_j\sim\mathcal N(\theta_j,\sigma_j^2).
$$

For the public estimand, the latent \(\theta_j\) are integrated out exactly:

$$
p(\mu,\tau\mid y)
\propto p(\mu)p(\tau)
\prod_{j=1}^{8}
\mathcal N\!\left(y_j;\mu,\tau^2+\sigma_j^2\right).
$$

The simulator therefore accepts only \((\mu,\tau)\) and draws each reported
effect from the corresponding marginal Gaussian. This is exactly the same
hyperparameter posterior as the hierarchical model above; it avoids treating
the eight latent school effects as estimands when they are not the object in
the comparison.

The independent reference integrates over \(\tau\) on a dense log grid and
integrates \(\mu\) analytically conditional on \(\tau\). Its deterministic
moments are

$$
\mathbb E[(\mu,\tau)\mid y]=(6.4720,\,4.7531),\qquad
\operatorname{SD}[(\mu,\tau)\mid y]=(4.1912,\,3.6838).
$$

Means, standard deviations, and correlation use deterministic quadrature
moments. Interval, marginal-CDF, and joint-grid comparisons use 400,000 fixed
draws from that quadrature law and are labeled as draw-based in the evidence.

The five public-API fits use 50,000 simulations each, ten diagonal-Gaussian
components, and seeds 73001--73005:

| Seed | `mu` mean | `tau` mean | `mu` SD | `tau` SD | Max CDF error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 73001 | 6.5305 | 5.0521 | 4.2670 | 3.9206 | 0.03395 |
| 73002 | 6.1960 | 4.7005 | 4.1413 | 3.8387 | 0.02873 |
| 73003 | 6.0483 | 4.5063 | 4.0678 | 3.6340 | 0.04246 |
| 73004 | 6.9624 | 4.6186 | 4.3167 | 3.7311 | 0.04997 |
| 73005 | 6.3516 | 4.6437 | 4.1008 | 3.7344 | 0.02047 |

Worst across the five runs:

| Diagnostic | Result | Limit |
| --- | ---: | ---: |
| Posterior-mean error, in exact posterior SDs | 0.11701 | 0.15 |
| Relative posterior-SD error | 0.06428 | 0.15 |
| 95% interval-endpoint error, in exact posterior SDs | 0.18910 | 0.20 |
| Marginal-CDF supremum error | 0.04997 | 0.10 |
| Correlation error | 0.04284 | 0.15 |
| Joint 20-by-20 quantile-grid total variation | 0.06350 | 0.08 |

Run the checked comparison from a repository checkout:

```bash
python -m pip install ".[replication]"
python replication/eight_schools/run_validation.py --quiet
```

The committed compact evidence records the exact configuration, fixed seeds,
per-seed results, and model fingerprints. Code and evidence are in
[`replication/eight_schools`](https://github.com/AndrewBai-Marketing/NPE_Econ/tree/main/replication/eight_schools).

The marginalized representation and training budget were chosen while
repairing the earlier benchmark. The five-seed result is reproducible
development evidence, not a preregistered holdout or a simulation-based
calibration campaign. The joint metric is a coarsened finite-grid diagnostic,
not a proof of exact joint-law recovery. In particular, dependence is weak in
this posterior, and the current joint-TV and correlation limits would not
reject a product-of-exact-marginals negative control. Dependence recovery is
therefore not established by this benchmark.

The earlier checked-in centered ten-dimensional run was only a permissive
pipeline smoke test. Its reference column also reported finite reference-draw
means as though they were deterministic quadrature means. That table has been
replaced rather than relabeled as successful accuracy evidence.

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
provides a separate, direct five-seed public-API accuracy comparison for a
two-parameter marginalized hyperposterior.

None of these results establishes structural identification, correct model
specification, causal interpretation, or universal posterior accuracy. A new
application needs its own comparator, simulation-based calibration, or other
predeclared validation study.
