# Generated from examples/03_custom_simulator.py by scripts/render_readme_examples.py.
# Edit the executable example, then rerun this script with --write.

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
