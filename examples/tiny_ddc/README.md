# Tiny DDC legacy CLI example

This retained compatibility example exercises the project-file CLI and the
legacy finite-grid estimator. It is tested from an installed wheel, but it is
not the primary public-beta `StructuralModel -> fit -> infer` workflow.

```bash
structnpe simulate --config config.yaml
structnpe train --config config.yaml
structnpe infer --config config.yaml --observed observed_example.csv
```
