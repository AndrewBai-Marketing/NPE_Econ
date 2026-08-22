# Generated from examples/01_quickstart_saved_estimator.py by scripts/render_readme_examples.py.
# Edit the executable example, then rerun this script with --write.

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
