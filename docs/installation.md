# Installation

`structnpe` 0.1.0b1 supports Python 3.11–3.13, inclusive. Python 3.10 and 3.14
are outside the declared beta range. Install from a reviewed source
checkout or from a locally built wheel; this release task does not publish
anything to PyPI.

## Inference-only installation

The base package is sufficient to load a completed beta estimator and draw
from its MDN on CPU. The saved network is evaluated with NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Training installation

Training the documented estimator requires Torch:

```bash
python -m pip install ".[neural]"
```

CPU is the supported default. If you need an accelerator, install a Torch
build appropriate to that machine and request the device explicitly. Do not
assume an accelerator run is bit-for-bit deterministic.

## Development installation

Contributors who need the test/build tools and the neural trainer should use:

```bash
python -m pip install -e ".[dev,neural]"
```

Run development checks on one of the supported Python versions. The editable
install is for development only; release verification must use the built wheel
and source distribution in fresh environments.

## Local verification

```bash
python -c "import structnpe; print(structnpe.__version__)"
```

For release review, build and test the wheel and source distribution in fresh
temporary environments. A successful import from the source tree is not a
substitute for a clean-wheel test.

Legacy alpha project files may require their original local simulator code.
They are trusted-local inputs and are not interchangeable with the
checksummed 0.1 beta artifact format.
