# Generated from examples/02_bayesian_workflow.py by scripts/render_readme_examples.py.
# Edit the executable example, then rerun this script with --write.

summary = result.summary()
draws = result.to_dataframe()
names = list(estimator.parameter_names)
covariance = pd.DataFrame(result.covariance(), index=names, columns=names)
correlation = pd.DataFrame(result.correlation(), index=names, columns=names)

probability = float((draws["maintenance_cost"] > 0.4).mean())
cost_ratio = draws["replacement_cost"] / draws["maintenance_cost"]
ratio_interval = cost_ratio.quantile([0.025, 0.5, 0.975])
