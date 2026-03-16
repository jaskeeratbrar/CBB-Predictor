"""Prediction model for CBB game outcomes.

Starts with calibrated LogisticRegression. Ensemble with XGBoost added later.
Supports Bayesian posterior updating between weekly retrains.
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score

import db
from features import FeatureEngine

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class PredictionModel:
    """Calibrated ensemble model for predicting CBB game outcomes."""

    def __init__(self, version: str = None):
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lr: CalibratedClassifierCV = None
        self.xgb = None  # Added in Phase 5
        self.feature_names: list = FeatureEngine.FEATURE_NAMES
        self.ensemble_weights = {"lr": 1.0}  # Start with LR only
        self.prior_alpha = 10.0  # Beta prior for Bayesian updates
        self.prior_beta = 10.0

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train model on historical data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (1=home win, 0=away win).

        Returns:
            Metrics dict: accuracy, brier_score, log_loss, cv_mean, cv_std.
        """
        log.info("Training model v%s on %d samples, %d features", self.version, X.shape[0], X.shape[1])

        # Calibrated logistic regression
        base_lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self.lr = CalibratedClassifierCV(base_lr, cv=5, method="sigmoid")
        self.lr.fit(X, y)

        # Metrics on training set (for logging; real eval is CV)
        y_prob = self.lr.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "brier_score": round(brier_score_loss(y, y_prob), 4),
            "log_loss": round(log_loss(y, y_prob), 4),
        }

        # Cross-validation on base LR for unbiased estimate
        cv_scores = cross_val_score(base_lr, X, y, cv=5, scoring="accuracy")
        metrics["cv_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_std"] = round(cv_scores.std(), 4)

        log.info("Training complete: %s", metrics)

        # Save model version to DB
        db.save_model_version(
            self.version,
            metrics["cv_mean"],
            metrics["brier_score"],
            metrics["log_loss"],
            f"Trained on {X.shape[0]} samples",
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated P(home_win) for each row.

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities.
        """
        if self.lr is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        probs = self.lr.predict_proba(X)[:, 1]

        # Future: ensemble with XGBoost
        # if self.xgb is not None:
        #     xgb_probs = self.xgb.predict_proba(X)[:, 1]
        #     probs = (self.ensemble_weights["lr"] * probs +
        #              self.ensemble_weights["xgb"] * xgb_probs)

        return probs

    def predict_single(self, features: dict) -> float:
        """Predict P(home_win) for a single game.

        Args:
            features: Dict of feature_name -> value.

        Returns:
            Float probability.
        """
        engine = FeatureEngine()
        X = engine.features_to_array(features)
        return float(self.predict(X)[0])

    def bayesian_update(self, prior_prob: float, outcome: int) -> float:
        """Update probability using Beta-Binomial conjugate model.

        Between weekly retrains, this adjusts the model's confidence
        based on observed outcomes for similar matchups.

        Args:
            prior_prob: Model's pre-game probability estimate.
            outcome: 1 if home won, 0 if away won.

        Returns:
            Updated posterior probability.
        """
        # Convert prior probability to Beta parameters
        alpha = self.prior_alpha * prior_prob
        beta = self.prior_beta * (1 - prior_prob)

        # Update with observation
        alpha += outcome
        beta += (1 - outcome)

        # Posterior mean
        posterior = alpha / (alpha + beta)
        return round(posterior, 4)

    def save(self) -> str:
        """Serialize model to models/{version}.pkl."""
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / f"{self.version}.pkl"
        state = {
            "version": self.version,
            "lr": self.lr,
            "xgb": self.xgb,
            "feature_names": self.feature_names,
            "ensemble_weights": self.ensemble_weights,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info("Model saved to %s", path)
        return str(path)

    @classmethod
    def load(cls, version: str) -> "PredictionModel":
        """Load a saved model by version string."""
        path = MODELS_DIR / f"{version}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(version=state["version"])
        model.lr = state["lr"]
        model.xgb = state.get("xgb")
        model.feature_names = state["feature_names"]
        model.ensemble_weights = state["ensemble_weights"]
        model.prior_alpha = state.get("prior_alpha", 10.0)
        model.prior_beta = state.get("prior_beta", 10.0)
        log.info("Loaded model v%s", version)
        return model

    @classmethod
    def load_latest(cls) -> "PredictionModel":
        """Load the most recently saved model."""
        MODELS_DIR.mkdir(exist_ok=True)
        pkls = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        if not pkls:
            raise FileNotFoundError("No saved models found in models/")
        latest = pkls[-1].stem
        return cls.load(latest)

    def retrain(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Full retrain with new data. Bumps version."""
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = self.train(X, y)
        self.save()
        return metrics
