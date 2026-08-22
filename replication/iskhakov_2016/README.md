# Iskhakov et al. (2016)-style bounded Monte Carlo

This directory compares the independently written NFXP comparator with one
trained `structnpe` estimator across repeated simulated Rust panels. It is a
bounded smoke design, not a reproduction of the paper's complete NFXP/MPEC
table.

The frozen data-generating process has 175 states, 50 buses, 120 periods,
linear cost scale `0.001`, true costs `(RC, slope) = (11.7257, 2.4569)`, and
transition probabilities `(0.0937, 0.4475, 0.4459, 0.0127, 0.0002)`. The full
prospective campaign covers betas `(0.975, 0.985, 0.995, 0.999, 0.9995,
0.9999)` and 250 panels per beta.

[`full_campaign.json`](full_campaign.json) records that complete design and is
explicitly marked `configured_not_run`. The runner refuses that scale without
`--confirm-full-campaign`. [`smoke_config.json`](smoke_config.json) records the
small default. The path-sanitized completed smoke evidence is
[`expected/smoke_metrics.json`](expected/smoke_metrics.json).

Run a bounded comparison:

```bash
python -m replication.iskhakov_2016.run_monte_carlo \
  --betas 0.975 \
  --repetitions 3 \
  --simulations 300 \
  --epochs 3 \
  --draws 300 \
  --quiet \
  --output replication/iskhakov_2016/results/smoke.json
```

The JSON output separates simulation/training time, loaded NPE inference time,
and NFXP time. For each method it reports bias and RMSE; NPE additionally
reports 95% interval coverage and posterior correlation. The amortization
break-even is computed only when measured per-panel NPE inference is faster.

The completed three-panel beta-0.975 smoke produced NFXP bias
`(-1.128, -0.274)` and RMSE `(1.605, 0.461)`, versus NPE posterior-mean bias
`(0.276, 0.154)` and RMSE `(0.369, 0.157)`. Both nominal 95% coverage values
were 1.0. NFXP took 1.364 seconds per panel; loaded NPE inference took 0.000406
seconds per panel after 8.65 seconds of simulation and 66.35 seconds of
training. The resulting descriptive break-even was 55 panels. With only three
replications, all bias, coverage, and timing comparisons are extremely noisy
and are not treated as performance evidence.

Important differences from the published campaign:

- transition probabilities are fixed at their declared DGP values for both
  NFXP and NPE rather than re-estimated;
- MPEC is not implemented and is reported as unsupported;
- the optimizer, prior, neural approximation, and smoke budgets differ;
- small-run coverage is descriptive and very noisy;
- only an actually executed six-beta/250-run output may be described as a full
  campaign.

Primary references:

- [Iskhakov, Lee, Rust, Schjerning, and Seo (2016), DOI 10.3982/ECTA12605](https://doi.org/10.3982/ECTA12605)
- [Official authors' source repository](https://github.com/fediskhakov/MPECvsNFXP)
- [Public ruspy replication and design notes](https://ruspy.readthedocs.io/en/latest/tutorials/replication/replication_iskhakov_et_al_2016.html)
