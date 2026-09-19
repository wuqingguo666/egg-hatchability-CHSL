from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from src.config import STACKING_INTERNAL_CV


def build_random_forest() -> RandomForestRegressor:
    # Hyperparameters used by the final RF evaluation code.
    return RandomForestRegressor(n_estimators=1000, random_state=0)


def build_lightgbm() -> LGBMRegressor:
    return LGBMRegressor()


def build_svr() -> SVR:
    return SVR()


def build_hsl() -> StackingRegressor:
    """Two-stage HSL: heterogeneous stage-0 models and linear stage-1 model."""
    base_models = [
        ("rf", RandomForestRegressor()),
        ("svr", SVR()),
        ("lgb", LGBMRegressor()),
    ]
    return StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=STACKING_INTERNAL_CV,
    )


def baseline_model_builders():
    return {
        "RF": build_random_forest,
        "LightGBM": build_lightgbm,
        "SVR": build_svr,
        "HSL": build_hsl,
    }

