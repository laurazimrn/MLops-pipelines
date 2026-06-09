import json
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import pipeline.train as t


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "tenure":          rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "total_charges":   rng.uniform(100, 8000, n).round(2),
        "num_products":    rng.integers(1, 6, n),
        "support_calls":   rng.integers(0, 10, n),
        "has_contract":    rng.integers(0, 2, n),
        "is_senior":       rng.integers(0, 2, n),
        "has_partner":     rng.integers(0, 2, n),
        "churn":           rng.integers(0, 2, n),
    })


@pytest.fixture
def fitted_model(small_df):
    X = small_df.drop(columns=["churn"])
    y = small_df["churn"]
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    return clf, X, y


def test_evaluate_keys_and_range(fitted_model):
    model, X, y = fitted_model
    metrics = t.evaluate(model, X, y)
    assert set(metrics) == {"auc_roc", "f1"}
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_save_artifacts_creates_files(tmp_path, monkeypatch, fitted_model):
    model, X, _ = fitted_model
    monkeypatch.setattr(t, "MODEL_DIR",    tmp_path / "models")
    monkeypatch.setattr(t, "METRICS_PATH", tmp_path / "metrics/scores.json")

    features = list(X.columns)
    metrics  = {"auc_roc": 0.85, "f1": 0.72}
    t.save_artifacts(model, features, metrics)

    assert (tmp_path / "models" / "model.joblib").exists()
    saved_features = json.loads((tmp_path / "models" / "features.json").read_text())
    assert saved_features["features"] == features
    saved_metrics = json.loads((tmp_path / "metrics" / "scores.json").read_text())
    assert saved_metrics == metrics


def test_load_data_shape_and_no_leakage(tmp_path, monkeypatch, small_df):
    csv_path = tmp_path / "churn.csv"
    small_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(t, "DATA_PATH", csv_path)

    X, y = t.load_data()
    assert "churn" not in X.columns
    assert len(X) == len(y) == len(small_df)


def test_train_runs_and_saves(tmp_path, monkeypatch, small_df):
    import mlflow
    csv_path = tmp_path / "churn.csv"
    small_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(t, "DATA_PATH",    csv_path)
    monkeypatch.setattr(t, "MODEL_DIR",    tmp_path / "models")
    monkeypatch.setattr(t, "METRICS_PATH", tmp_path / "metrics/scores.json")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

    t.train(n_estimators=10, max_depth=3)

    assert (tmp_path / "models" / "model.joblib").exists()
    metrics = json.loads((tmp_path / "metrics" / "scores.json").read_text())
    assert metrics["auc_roc"] > 0.5
