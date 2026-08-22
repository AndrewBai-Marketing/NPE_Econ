# Exact posterior validation

Status: **PASS**

The target is the analytic posterior under the declared Normal prior,
Normal simulator, and sample-mean representation. Passing this benchmark
does not establish structural identification or correct specification.

| metric | value | maximum | pass |
| --- | ---: | ---: | --- |
| posterior_mean_mae | 0.011477428 | 0.1 | True |
| posterior_sd_mae | 0.006917082 | 0.07 | True |
| posterior_variance_mae | 0.0030779114 | 0.04 | True |
| central_95_endpoint_mae | 0.022779606 | 0.15 | True |
| marginal_quantile_grid_l1_mean | 0.014160332 | 0.1 | True |
| coverage_95_abs_error | 0.01875 | 0.2 | True |
| predictive_sd_mae | 0.0048784484 | 0.08 | True |

Threshold SHA-256: `676aeab7d484fc09c40cba5dc0eb433558afd50284798c36233366b9103d6349`

Metrics are generated only when this script is explicitly run.

## Simulation-based calibration smoke summary

| region | n | bias | RMSE | cov. 50 | cov. 80 | cov. 90 | cov. 95 | mean rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 32 | -0.0200282 | 0.225784 | 0.469 | 0.812 | 0.844 | 0.969 | 0.532 |
| lower | 11 | -0.000108953 | 0.157883 | 0.545 | 1.000 | 1.000 | 1.000 | 0.500 |
| middle | 10 | 0.013827 | 0.196632 | 0.500 | 0.900 | 0.900 | 1.000 | 0.494 |
| upper | 11 | -0.0707249 | 0.297026 | 0.364 | 0.545 | 0.636 | 0.909 | 0.598 |

Coverage and ranks are descriptive at this finite case count and are conditional on the declared prior, simulator, representation, and fitted approximation.
