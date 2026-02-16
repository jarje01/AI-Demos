"""
Model Training – trains, evaluates, and persists the ML prediction model.
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import MODEL_CONFIG, PATHS
from features import get_feature_columns

logger = logging.getLogger(__name__)


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: dict | None = None):
    """Instantiate the ML model specified in MODEL_CONFIG."""
    cfg = cfg or MODEL_CONFIG
    mtype = cfg["model_type"].lower()

    if mtype == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            class_weight="balanced",
        )

    elif mtype == "gradient_boost":
        return GradientBoostingClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
            subsample=0.8,
        )

    elif mtype == "xgboost":
        if not HAS_XGB:
            logger.warning("xgboost not installed – falling back to RandomForest.")
            return build_model({**cfg, "model_type": "random_forest"})
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["xgb_learning_rate"],
            subsample=cfg["xgb_subsample"],
            colsample_bytree=cfg["xgb_colsample_bytree"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
            eval_metric="mlogloss",
            verbosity=0,
        )

    else:
        raise ValueError(f"Unknown model_type: {mtype}")


# ─── Training pipeline ────────────────────────────────────────────────────────

class FXModel:
    """Wrapper that pairs a sklearn estimator with its StandardScaler."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or MODEL_CONFIG
        self.model = build_model(self.cfg)
        self.base_model = None
        self.is_calibrated = False
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.is_fitted = False

    # ── Fit ──
    def fit(self, df_features: pd.DataFrame) -> "FXModel":
        self.feature_cols = get_feature_columns(df_features)
        X = df_features[self.feature_cols]
        y = df_features["target_class"].values

        # Drop any residual NaNs
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        logger.info("Training on %d samples, %d features …", len(X), X.shape[1])
        X_scaled = self.scaler.fit_transform(X)

        enable_calibration = bool(self.cfg.get("enable_calibration", False))
        calibration_fraction = float(self.cfg.get("calibration_fraction", 0.2))
        min_calibration_samples = int(self.cfg.get("min_calibration_samples", 120))

        if enable_calibration and 0.0 < calibration_fraction < 0.5:
            n_cal_target = int(len(X_scaled) * calibration_fraction)
            can_calibrate = (
                n_cal_target >= min_calibration_samples
                and len(np.unique(y)) >= 2
            )

            if can_calibrate:
                cal_method = self.cfg.get("calibration_method", "sigmoid")
                n_splits = int(self.cfg.get("calibration_cv_splits", 3))
                n_splits = max(2, min(n_splits, len(X_scaled) - 1))
                cv = TimeSeriesSplit(n_splits=n_splits)

                try:
                    calibrator = CalibratedClassifierCV(
                        estimator=build_model(self.cfg),
                        method=cal_method,
                        cv=cv,
                    )
                except TypeError:
                    calibrator = CalibratedClassifierCV(
                        base_estimator=build_model(self.cfg),
                        method=cal_method,
                        cv=cv,
                    )

                calibrator.fit(X_scaled, y)
                self.model = calibrator
                self.base_model = build_model(self.cfg)
                self.base_model.fit(X_scaled, y)
                self.is_calibrated = True
                logger.info(
                    "Training complete with probability calibration (%s, ts-cv=%d).",
                    cal_method,
                    n_splits,
                )
            else:
                self.model = build_model(self.cfg)
                self.model.fit(X_scaled, y)
                self.base_model = self.model
                self.is_calibrated = False
                logger.info("Calibration skipped due to insufficient samples.")
                logger.info("Training complete.")
        else:
            self.model = build_model(self.cfg)
            self.model.fit(X_scaled, y)
            self.base_model = self.model
            self.is_calibrated = False
            logger.info("Training complete.")

        self.is_fitted = True
        return self

    # ── Predict ──
    def predict(self, df_features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X = df_features[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Returns probability array (shape: n_samples × n_classes)."""
        self._check_fitted()
        X = df_features[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_signal_probabilities(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [prob_short, prob_flat, prob_long]
        aligned to df_features.index. Classes are sorted: -1, 0, 1.
        """
        self._check_fitted()
        proba = self.predict_proba(df_features)
        classes = self.model.classes_           # e.g. [-1, 0, 1]

        col_map = {-1: "prob_short", 0: "prob_flat", 1: "prob_long"}
        result = pd.DataFrame(index=df_features.index)
        for i, cls in enumerate(classes):
            result[col_map[cls]] = proba[:, i]

        # Ensure all three columns exist
        for col in ["prob_short", "prob_flat", "prob_long"]:
            if col not in result.columns:
                result[col] = 0.0

        return result

    # ── Evaluate ──
    def evaluate(self, df_features: pd.DataFrame) -> dict:
        self._check_fitted()
        X = df_features[self.feature_cols]
        y_true = df_features["target_class"].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_true))
        X, y_true = X[mask], y_true[mask]

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

        try:
            proba = self.model.predict_proba(X_scaled)
            if proba.shape[1] >= 2:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro"
                )
        except Exception:
            pass

        logger.info("Accuracy: %.3f", metrics["accuracy"])
        return metrics

    def cross_validate(self, df_features: pd.DataFrame, cv: int = 5) -> dict:
        """Time-series-safe cross-validation (uses TimeSeriesSplit internally)."""
        self._check_fitted()  # ensures scaler is fitted

        X = df_features[self.feature_cols]
        y = df_features["target_class"].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        X_scaled = self.scaler.transform(X)
        tscv = TimeSeriesSplit(n_splits=cv)

        cv_model = build_model(self.cfg)
        scores = cross_val_score(
            cv_model, X_scaled, y, cv=tscv, scoring="accuracy", n_jobs=-1
        )
        return {"cv_accuracy_mean": scores.mean(), "cv_accuracy_std": scores.std(), "cv_scores": scores.tolist()}

    # ── Feature importance ──
    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        estimator = self.base_model if self.base_model is not None else self.model
        if not hasattr(estimator, "feature_importances_"):
            return pd.Series(dtype=float)
        return pd.Series(
            estimator.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

    # ── Persistence ──
    def save(self, path: str | None = None) -> str:
        os.makedirs(PATHS["models_dir"], exist_ok=True)
        path = path or os.path.join(PATHS["models_dir"], "fx_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "FXModel":
        path = path or os.path.join(PATHS["models_dir"], "fx_model.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return obj

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import download_data, split_data
    from features import build_features

    df = download_data()
    train_raw, test_raw = split_data(df)
    train_feat = build_features(train_raw)
    test_feat = build_features(test_raw)

    model = FXModel()
    model.fit(train_feat)
    metrics = model.evaluate(test_feat)
    print("\n=== OOS Evaluation ===")
    print(metrics["report"])

    top_fi = model.feature_importance().head(10)
    print("\n=== Top-10 Feature Importances ===")
    print(top_fi.to_string())

    model.save()
