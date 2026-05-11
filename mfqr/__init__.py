"""
mfqr — Multi-Fidelity Quantile Regression library.

Public API:
    build_methods(...)               # construct a dict of TransferQuantileRegressor instances
    TransferQuantileRegressor(...)   # the estimator class
    standardise_separately(...)      # in-place data standardisation
    conformal_calibrate(...)         # CQR margin q
    create_mf_split(...)             # 12-section disjoint split
    ConditionalCDF_QRF, ConditionalCDF_RFFGP   # backend conditional CDF estimators

The TransferQuantileRegressor supports two one-step correction spaces:
    one_step_space="residual"  (default) — HF correction CDF is fit on
        offset-subtracted residuals.  Used by Sections 5.1, 5.2, 5.4 of the paper.
    one_step_space="raw"                 — HF correction CDF is fit on the raw
        HF response.  Used by Section 5.2.3 ("Misinformative regime"), where the
        bias is in the affine offset itself, so operating in residual space
        would propagate that bias.
"""
from .core import (
    TransferQuantileRegressor,
    build_methods,
)
from .data_utils import (
    standardise_separately,
    conformal_calibrate,
    create_mf_split,
)
from .cdf_models import (
    ConditionalCDF_QRF,
    ConditionalCDF_RFFGP,
)

__all__ = [
    "TransferQuantileRegressor",
    "build_methods",
    "standardise_separately",
    "conformal_calibrate",
    "create_mf_split",
    "ConditionalCDF_QRF",
    "ConditionalCDF_RFFGP",
]
