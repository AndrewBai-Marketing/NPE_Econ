# Custom Simulator Template

This template is a runnable toy project. Replace `model.py` with your own
simulator-defined structural model.

The posterior target is conditional on the summary returned by
`MyModel.summarize`, not automatically on the full raw dataset.

```bash
structnpe simulate --config config.yaml
structnpe train --config config.yaml
structnpe infer --config config.yaml --observed observed_example.csv
structnpe validate --config config.yaml --observed observed_example.csv
structnpe counterfactual --config config.yaml --observed observed_example.csv --policy policy.yaml
```
