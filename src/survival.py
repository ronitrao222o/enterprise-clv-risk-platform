"""Regularized Cox model with training-only sklearn preprocessing and Kaplan–Meier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES


@dataclass
class SurvivalModel:
    """One independent landmark per customer; frozen covariates after the landmark."""

    penalizer: float = 0.1
    preprocessor: ColumnTransformer | None = None
    cox: CoxPHFitter | None = None
    km: KaplanMeierFitter | None = None
    columns: list[str] = field(default_factory=list)
    max_horizon: int = 0

    def fit(self, labeled: pd.DataFrame) -> "SurvivalModel":
        if labeled.customer_id.duplicated().any():
            raise ValueError("Training requires one landmark per customer.")
        if labeled.event_observed.sum() < 10:
            raise ValueError(
                "Too few churn events to fit a reliable model; generate more accounts."
            )
        self.preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "category",
                    OneHotEncoder(
                        categories=CATEGORY_LEVELS,
                        drop="first",
                        handle_unknown="error",
                        sparse_output=False,
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ],
            verbose_feature_names_out=False,
        )
        values = self.preprocessor.fit_transform(labeled)
        names = self.preprocessor.get_feature_names_out()
        # A constant column cannot contribute to a Cox likelihood.
        self.columns = names[np.std(values, axis=0) > 1e-8].tolist()
        x = pd.DataFrame(values, columns=names, index=labeled.index)[self.columns]
        x["duration"] = labeled.duration
        x["event_observed"] = labeled.event_observed
        self.cox = CoxPHFitter(penalizer=self.penalizer)
        self.cox.fit(x, duration_col="duration", event_col="event_observed")
        self.km = KaplanMeierFitter(label="Training landmark KM").fit(
            labeled.duration, labeled.event_observed
        )
        self.max_horizon = int(labeled.duration.max())
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("Fit the model before predicting.")
        values = self.preprocessor.transform(features)
        return pd.DataFrame(
            values, columns=self.preprocessor.get_feature_names_out(), index=features.index
        )[self.columns]

    def predict(self, features: pd.DataFrame, horizon: int = 24) -> np.ndarray:
        if self.cox is None or horizon > self.max_horizon or horizon < 1:
            raise ValueError(
                f"Prediction horizon must be within observed training support (1..{self.max_horizon})."
            )
        curve = self.cox.predict_survival_function(
            self.transform(features), times=np.arange(1, horizon + 1)
        )
        return np.clip(curve.to_numpy().T, 0, 1)

    def contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Exact additive log partial-hazard terms relative to training mean, not causal effects."""
        x = self.transform(features)
        return (x - self.cox._norm_mean) * self.cox.params_
