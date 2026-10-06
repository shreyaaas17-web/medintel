import os
import joblib
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

os.makedirs("models", exist_ok=True)


def load_diabetes():
    d = fetch_openml(data_id=37, as_frame=True, parser="auto")  # Pima diabetes
    return d.data, (d.target.astype(str) == "tested_positive").astype(int)


def load_heart():
    d = fetch_openml(data_id=53, as_frame=True, parser="auto")  # heart-statlog
    return d.data, (d.target.astype(str) == "present").astype(int)


def load_breast():
    d = load_breast_cancer(as_frame=True)
    return d.data, (d.target == 0).astype(int)  # 1 = malignant


# To add a disease: add an entry here with a loader returning (X, y) where y=1 means disease
DISEASES = {
    "Diabetes": load_diabetes,
    "Heart Disease": load_heart,
    "Breast Cancer": load_breast,
}

for name, loader in DISEASES.items():
    X, y = loader()
    X = X.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, model.predict(X_te))
    auc = roc_auc_score(y_te, proba)
    print(f"{name}: accuracy={acc:.3f}  ROC-AUC={auc:.3f}")

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "min": X.min().to_dict(),
            "max": X.max().to_dict(),
            "mean": X.mean().to_dict(),
            "accuracy": acc,
            "auc": auc,
        },
        f"models/{name.replace(' ', '_').lower()}.joblib",
    )

print("All models saved in models/")