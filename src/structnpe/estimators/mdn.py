"""MDN estimator placeholder for research-alpha builds."""

from __future__ import annotations

import warnings

from .gaussian import GaussianPosterior


class MDNPosterior(GaussianPosterior):
    """Alpha MDN interface.

    The research repo contains neural posterior code in ``ddc_npe``. The
    standalone ``structnpe`` alpha keeps the public interface stable but uses a
    Gaussian fallback unless a project-specific torch MDN implementation is
    added. Reports should state this limitation.
    """

    estimator_type = "mdn"

    def __init__(self, ridge: float = 1.0e-6) -> None:
        warnings.warn(
            "The project-file posterior type 'mdn' is a legacy Gaussian placeholder. "
            "Use structnpe.fit(...) for the public beta's real neural MDN.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(ridge=ridge)
