import json, argparse
from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.datasets import load_iris

DATA_PATH    = Path("data/churn.csv")
MODEL_DIR    = Path("models")
METRICS_PATH = Path("metrics/scores.json")

def gerar_dados(path, n=1000):
    path.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "tenure":          rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "total_charges":   rng.uniform(100, 8000, n).round(2),
        "num_products":    rng.integers(1, 6, n),
        "support_calls":   rng.integers(0, 10, n),
        "has_contract":    rng.integers(0, 2, n),
        "is_senior":       rng.integers(0, 2, n),
        "has_partner":     rng.integers(0, 2, n),
    })
    prob = (
        0.6 * (df["tenure"] < 12).astype(float)
        + 0.3 * (df["support_calls"] > 5).astype(float)
        + 0.3 * (df["has_contract"] == 0).astype(float)
    )
    prob = (prob - prob.min()) / (prob.max() - prob.min())
    df["churn"] = (rng.uniform(0, 1, n) < prob).astype(int)
    df.to_csv(path, index=False)
    print(f"[INFO] Dataset gerado: {n} registros")

def avaliar(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "f1":      round(f1_score(y_test, y_pred), 4),
    }

def train(n_estimators=100, max_depth=5):
    if not DATA_PATH.exists():
        gerar_dados(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=["churn"])
    y  = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("churn-prediction")

    modelos = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
    }

    melhor_auc   = 0
    melhor_model = None
    melhor_nome  = None

    for nome, modelo in modelos.items():
        with mlflow.start_run(run_name=nome):
            modelo.fit(X_train, y_train)
            metrics = avaliar(modelo, X_test, y_test)

            mlflow.log_param("modelo", nome)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(modelo, "model")

            print(f"  {nome}: AUC={metrics['auc_roc']} | F1={metrics['f1']}")

            if metrics["auc_roc"] > melhor_auc:
                melhor_auc   = metrics["auc_roc"]
                melhor_model = modelo
                melhor_nome  = nome

    # Salva o melhor modelo
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)

    joblib.dump(melhor_model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "features.json").write_text(json.dumps({"features": list(X.columns)}))
    METRICS_PATH.write_text(json.dumps({"best_model": melhor_nome, "auc_roc": melhor_auc}, indent=2))

    print(f"\n✓ Melhor modelo: {melhor_nome} (AUC={melhor_auc})")
    print(f"✓ Salvo em: {MODEL_DIR}/model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth",    type=int, default=5)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth)
