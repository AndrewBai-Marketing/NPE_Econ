"""Conditional spline flows adapted from the stockpiling research estimator.

Torch and nflows are imported lazily so legacy MDN artifacts retain NumPy-only
inference. Flow states use the package's checked numeric archives, not pickle.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from ._artifacts import ArtifactError


SPLINE_TYPE = "autoregressive_rational_quadratic_spline"


def validate_architecture(architecture: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    limits = {"input_dim": 8192, "theta_dim": 256, "hidden_dim": 1024,
              "depth": 4, "flow_layers": 8, "num_bins": 32}
    values = {}
    for key, maximum in limits.items():
        value = architecture.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
            raise ArtifactError(f"Spline architecture field {key!r} must be an integer in [1, {maximum}].")
        values[key] = int(value)
    if values["num_bins"] < 2:
        raise ArtifactError("Spline flows require at least two bins.")
    bound = architecture.get("tail_bound")
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not np.isfinite(bound) or not 0 < bound <= 100:
        raise ArtifactError("Spline tail_bound must be finite and in (0, 100].")
    if architecture.get("activation") != "silu" or architecture.get("nflows_version") != "0.14":
        raise ArtifactError("Unsupported spline activation or nflows artifact version.")
    if architecture.get("conditioner") != "made_with_context_skip_v1":
        raise ArtifactError("Unsupported spline conditioner.")
    # Bound allocations before constructing an architecture from an artifact.
    h, d, x = values["hidden_dim"], values["theta_dim"], values["input_dim"]
    per_layer = 4 * (h * (x + d + 1) + values["depth"] * h * (2 * h + x + 4)
                     + d * (3 * values["num_bins"] + 1) * (h + 1))
    if values["flow_layers"] * per_layer > 20_000_000:
        raise ArtifactError("Spline architecture exceeds the supported weight allocation budget.")
    # Last slot is the legacy MDN component count; spline callers do not use it.
    return x, d, h, values["depth"], 1


def make_flow(architecture: Mapping[str, Any], *, seed: int | None = None):
    validate_architecture(architecture)
    try:
        import torch
        from nflows import distributions, flows, transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("Spline training and inference require structnpe[neural] (Torch and nflows).") from exc
    import importlib.metadata
    if importlib.metadata.version("nflows") != "0.14":
        raise RuntimeError("This spline artifact format requires nflows==0.14.")

    class ConditionalMADE(torch.nn.Module):
        """Give every output a context path, including the first coordinate.

        nflows 0.14 MADE's first output has no hidden-unit connections for
        dimension > 1. Context entering only those hidden units therefore
        cannot condition that output. This context-only skip preserves the
        autoregressive restrictions on parameter inputs.
        """

        def __init__(self, base):
            super().__init__()
            self.base = base
            self.context_net = torch.nn.Sequential(
                torch.nn.Linear(int(architecture["input_dim"]), int(architecture["hidden_dim"])),
                torch.nn.SiLU(),
                torch.nn.Linear(int(architecture["hidden_dim"]),
                                int(architecture["theta_dim"]) * (3 * int(architecture["num_bins"]) - 1)),
            )

        def forward(self, inputs, context=None):
            return self.base(inputs, context=context) + self.context_net(context)

    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
        layers = []
        for _ in range(int(architecture["flow_layers"])):
            layers.append(transforms.RandomPermutation(features=int(architecture["theta_dim"])))
            transform = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=int(architecture["theta_dim"]),
                hidden_features=int(architecture["hidden_dim"]),
                context_features=int(architecture["input_dim"]),
                num_bins=int(architecture["num_bins"]), tails="linear",
                tail_bound=float(architecture["tail_bound"]),
                num_blocks=int(architecture["depth"]), activation=torch.nn.functional.silu,
                dropout_probability=0.0, use_batch_norm=False,
            )
            transform.autoregressive_net = ConditionalMADE(transform.autoregressive_net)
            layers.append(transform)
        model = flows.Flow(transforms.CompositeTransform(layers),
                           distributions.StandardNormal([int(architecture["theta_dim"])]))
    model._structnpe_spline = True
    return torch, model.float()


def numpy_weights(model: Any) -> dict[str, np.ndarray]:
    return {f"flow_{i:04d}": value.detach().cpu().numpy().astype(np.float32)
            for i, (_, value) in enumerate(model.state_dict().items())}


def load_weights(model: Any, weights: Mapping[str, np.ndarray]) -> None:
    import torch
    state = {}
    for i, (name, template) in enumerate(model.state_dict().items()):
        values = np.asarray(weights[f"flow_{i:04d}"])
        if name.endswith("_permutation"):
            if not np.array_equal(np.sort(values), np.arange(len(values))):
                raise ArtifactError("Invalid spline permutation buffer.")
        if not template.is_floating_point() and np.any(values != np.round(values)):
            raise ArtifactError("Non-integral spline index buffer.")
        if name.endswith((".mask", ".degrees")) and not np.array_equal(values, template.cpu().numpy()):
            raise ArtifactError("Spline autoregressive mask or degrees disagree with the architecture.")
        state[name] = torch.as_tensor(values.copy(), dtype=template.dtype, device=template.device)
    model.load_state_dict(state, strict=True)


@lru_cache(maxsize=32)
def _cached_shapes(serialized: str):
    _, model = make_flow(json.loads(serialized), seed=0)
    return tuple((f"flow_{i:04d}", tuple(value.shape))
                 for i, value in enumerate(model.state_dict().values()))


def weight_shapes(architecture: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    validate_architecture(architecture)
    return dict(_cached_shapes(json.dumps(dict(architecture), sort_keys=True)))


def sample(estimator: Any, representation: np.ndarray, *, draws: int, seed: int, device: str) -> np.ndarray:
    torch, model = make_flow(estimator.architecture, seed=0)
    load_weights(model, estimator.weights)
    model.to(device).eval()
    context = (representation - estimator.x_mean) / estimator.x_scale
    if not np.all(np.isfinite(context)) or np.any(np.abs(context) > np.finfo(np.float32).max):
        raise ValueError("Observation is outside the spline's supported numeric range.")
    output = np.empty((len(context), draws, len(estimator.parameters)))
    cuda_devices = []
    if device.startswith("cuda"):
        cuda_index = torch.device(device).index
        cuda_devices = [torch.cuda.current_device() if cuda_index is None else cuda_index]
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.random.default_generator.manual_seed(seed)
        for cuda_index in cuda_devices:
            torch.cuda.default_generators[cuda_index].manual_seed(seed)
        for i, row in enumerate(context):
            inputs = torch.as_tensor(row[None, :], dtype=torch.float32, device=device)
            for start in range(0, draws, 1024):
                size = min(1024, draws - start)
                values = model.sample(size, context=inputs)[0].cpu().numpy().astype(float)
                output[i, start:start + size] = values * estimator.theta_scale + estimator.theta_mean
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Spline sampling produced non-finite transformed parameters.")
    return output
