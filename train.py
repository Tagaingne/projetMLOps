import os
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── 1. Chargement du dataset ──────────────────────────────────────────────────
print("Chargement du dataset Breast Cancer...")
data = load_breast_cancer()
X, y = data.data, data.target

print(f"  Nombre d'exemples : {X.shape[0]}")
print(f"  Nombre de features : {X.shape[1]}")
print(f"  Classes : {list(data.target_names)}")

# ── 2. Prétraitement ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 3. Entraînement ───────────────────────────────────────────────────────────
print("\nEntraînement du modèle (LogisticRegression)...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ── 4. Évaluation ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n── Métriques ────────────────────────────────")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  F1-score  : {f1:.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 5. Sauvegarde de l'artefact ───────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
}

with open("model/model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nModèle sauvegardé dans model/model.pkl")