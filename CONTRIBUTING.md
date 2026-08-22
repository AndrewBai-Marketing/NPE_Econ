# Contributing

`structnpe` 0.1 is a narrow public beta. Contributions should preserve its
documented simulator-to-posterior contract and claims discipline:

- do not include proprietary or private data;
- do not claim identification or simulator correctness;
- keep the primary beta path separate from trusted-local legacy APIs;
- preserve safe, checksummed, pickle-free beta artifacts;
- keep validation diagnostics visible;
- keep examples synthetic, small, and runnable;
- add tests for new public APIs or CLI commands.

Create a Python 3.11, 3.12, or 3.13 environment and install the development and
training dependencies:

```bash
python -m pip install -e ".[dev,neural]"
```

Before opening a pull request, run the package-focused checks:

```bash
python -m compileall -q src/structnpe
python -m pytest -q tests/test_structnpe_*.py
python -m build --wheel --sdist
```

Changes to serialization, loading, fingerprints, parameter transforms,
observation adapters, or device handling need focused tests for failure paths
as well as the successful path. Review the built wheel and source distribution
for accidental research, data, cache, or trained-model payloads.

Large experiments, private data, and paper artifacts belong outside the public
package surface. A passing software test or benchmark is not a scientific
validity claim.
