# Structural exact-grid validation

Status: **PASS**

The comparator is exact for the declared finite-grid prior and sufficient
state-action-count observation. Acceptance applies only to the one
predeclared midpoint-truth panel generated with the profile's fixed seed;
it is not evidence of amortized accuracy across observations, identification,
or specification validity for other structural models.

| metric | value | maximum | pass |
| --- | ---: | ---: | --- |
| parameter_mean_normalized_mae | 0.029457837 | 0.1 | True |
| parameter_sd_normalized_mae | 0.012536964 | 0.1 | True |
| central_95_endpoint_normalized_mae | 0.027307277 | 0.15 | True |
| covariance_normalized_max_abs_error | 0.0058668463 | 0.05 | True |
| correlation_max_abs_error | 0.0436754 | 0.25 | True |
| marginal_quantile_grid_l1_normalized_mean | 0.044690429 | 0.12 | True |
| joint_grid_voronoi_total_variation | 0.25206966 | 0.35 | True |
| policy_mean_abs_error | 0.0010231867 | 0.06 | True |
| policy_sd_abs_error | 0.0020284976 | 0.05 | True |
| policy_95_endpoint_mae | 0.004519661 | 0.08 | True |
| predictive_cell_probability_max_abs_error | 0.0022975736 | 0.06 | True |
| parameter_box_support_violation_rate | 0 | 0 | True |

Threshold SHA-256: `96cf03a5f80f9d7849ca8842f6aed232367907c9b02d67a8c657191161b78e92`

Metrics are generated only when this script is explicitly run.
