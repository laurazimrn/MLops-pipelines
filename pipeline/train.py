import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')


def train():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('accuracy', float(acc))

        joblib.dump(clf, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)

    print('Training complete. Model saved to', MODEL_PATH)


if __name__ == '__main__':
    train()
