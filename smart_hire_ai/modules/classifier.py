"""
SmartHire AI – ML Classifier
Train and predict candidate suitability using multiple ML models.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from utils.constants import ML_CONFIG, MODEL_PATH, MODELS_DIR


class SuitabilityClassifier:
    """ML classifier for predicting candidate suitability."""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.roc_data = {}

    def generate_synthetic_data(self, n_samples=None):
        """
        Generate realistic synthetic hiring data for training.
        Features: experience_score, education_score, num_skills_norm, num_skills
        Target: hired (0 or 1)
        """
        n = n_samples or ML_CONFIG["n_synthetic_samples"]
        np.random.seed(ML_CONFIG["random_state"])

        # Generate features
        experience = np.random.uniform(0, 1, n)
        education = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n)
        num_skills_norm = np.random.uniform(0, 1, n)
        num_skills = np.random.randint(1, 25, n)

        # Create realistic hiring probability
        hiring_prob = (
            0.35 * experience +
            0.25 * education +
            0.30 * num_skills_norm +
            0.10 * np.random.uniform(0, 1, n)  # Random factor
        )

        # Convert to binary outcome
        threshold = np.percentile(hiring_prob, 45)  # ~55% hire rate
        hired = (hiring_prob >= threshold).astype(int)

        X = np.column_stack([experience, education, num_skills_norm, num_skills])
        feature_names = ["experience_score", "education_score", "num_skills_norm", "num_skills"]

        return X, hired, feature_names

    def train_models(self, X=None, y=None):
        """
        Train all ML models. If no data provided, uses synthetic data.
        Returns dict of model metrics.
        """
        if X is None or y is None:
            X, y, _ = self.generate_synthetic_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=ML_CONFIG["test_size"],
            random_state=ML_CONFIG["random_state"],
            stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        model_instances = {
            "Logistic Regression": LogisticRegression(
                random_state=ML_CONFIG["random_state"], max_iter=1000
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=ML_CONFIG["random_state"]
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=ML_CONFIG["random_state"]
            ),
        }

        best_accuracy = 0

        for name, model in model_instances.items():
            # Train
            model.fit(X_train_scaled, y_train)
            self.models[name] = model

            # Predict
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            self.metrics[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": cm.tolist(),
            }

            self.roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(roc_auc, 4),
            }

            # Track best model
            if acc > best_accuracy:
                best_accuracy = acc
                self.best_model = model
                self.best_model_name = name

        self.is_trained = True
        self._save_model()
        return self.metrics

    def predict(self, features):
        """
        Predict suitability for candidate features.
        features: array-like of shape (n_features,) or (n_samples, n_features)
        Returns: prediction (0/1), probability
        """
        if not self.is_trained:
            self._load_model()
        if self.best_model is None:
            return 0, 0.0

        features = np.array(features).reshape(1, -1) if np.array(features).ndim == 1 else np.array(features)
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.best_model.predict(features_scaled)[0]
            probability = self.best_model.predict_proba(features_scaled)[0][1]
            return int(prediction), round(float(probability), 4)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0, 0.0

    def predict_batch(self, feature_matrix):
        """Predict suitability for multiple candidates."""
        results = []
        for features in feature_matrix:
            pred, prob = self.predict(features)
            results.append({"prediction": pred, "probability": prob})
        return results

    def _save_model(self):
        """Save the best model and scaler to disk."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            joblib.dump({
                "model": self.best_model,
                "model_name": self.best_model_name,
                "scaler": self.scaler,
                "metrics": self.metrics,
            }, MODEL_PATH)
        except Exception as e:
            print(f"Error saving model: {e}")

    def _load_model(self):
        """Load a previously saved model."""
        if os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.best_model = data["model"]
                self.best_model_name = data["model_name"]
                self.scaler = data["scaler"]
                self.metrics = data.get("metrics", {})
                self.is_trained = True
            except Exception as e:
                print(f"Error loading model: {e}")

    def get_metrics_dataframe(self):
        """Return model metrics as a pandas DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        rows = []
        for name, m in self.metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1 Score": m["f1_score"],
                "ROC AUC": m["roc_auc"],
            })
        return pd.DataFrame(rows)
