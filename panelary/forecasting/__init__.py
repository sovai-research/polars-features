from .automl import (
    auto_elastic_net,
    auto_knn,
    auto_lasso,
    auto_linear_model,
    auto_ridge,
)
from .censored import censored_model, zero_inflated_model
from .elite import elite
from .knn import knn
from .linear import (
    elastic_net,
    elastic_net_cv,
    lasso,
    lasso_cv,
    linear_model,
    ridge,
    ridge_cv,
)
from .naive import naive
from .snaive import snaive


def _missing(name: str, extra: str):
    """Return a placeholder that raises an actionable error when *called*.

    Previously these names were bound to an ``ImportError`` **instance**, which
    meant ``fc.lightgbm`` was a value rather than something that raised: the
    failure surfaced far from its cause, as a "not callable" TypeError. The
    hints were also wrong -- they advertised the extras ``lgb``, ``cat`` and
    ``xgb``, none of which exist (see ``_deps._MODULE_TO_EXTRA``), so a user
    who followed them got "no matches found" from pip.
    """

    def _raise(*_args, **_kwargs):
        raise ImportError(
            f"panelary.forecasting.{name} requires the optional "
            f"'{extra}' extra: pip install 'panelary[{extra}]'"
        )

    _raise.__name__ = name
    _raise.__doc__ = f"Unavailable: install the '{extra}' extra to use {name}."
    return _raise


try:
    from .lance import ann
except ImportError:
    ann = _missing("ann", "ann")

try:
    from .automl import auto_lightgbm
    from .lightgbm import flaml_lightgbm, lightgbm
except ImportError:
    # `lightgbm` needs only the booster; the FLAML-driven paths need `automl`.
    auto_lightgbm = _missing("auto_lightgbm", "automl")
    flaml_lightgbm = _missing("flaml_lightgbm", "automl")
    lightgbm = _missing("lightgbm", "lightgbm")

try:
    from .catboost import catboost
except ImportError:
    catboost = _missing("catboost", "catboost")

try:
    from .xgboost import xgboost
except ImportError:
    xgboost = _missing("xgboost", "xgboost")


__all__ = [
    "ann",
    "auto_elastic_net",
    "auto_knn",
    "auto_lasso",
    "auto_lightgbm",
    "auto_linear_model",
    "auto_ridge",
    "catboost",
    "censored_model",
    "elastic_net_cv",
    "elastic_net",
    "elite",
    "flaml_lightgbm",
    "knn",
    "lasso_cv",
    "lasso",
    "lightgbm",
    "linear_model",
    "naive",
    "ridge_cv",
    "ridge",
    "snaive",
    "xgboost",
    "zero_inflated_model",
]
