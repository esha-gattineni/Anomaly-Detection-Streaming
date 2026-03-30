import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Features used for anomaly detection ──────────────────────
FEATURES = ['cpu_usage', 'memory_usage', 'latency_ms', 'error_rate']

def train():
    # ── 1. Load data ──────────────────────────────────────────
    df = pd.read_csv('data/telemetry_data.csv')
    print(f"Loaded {len(df)} events")

    # ── 2. Split train / test ─────────────────────────────────
    # Train only on normal data — Isolation Forest learns what normal looks like
    normal_df = df[df['is_anomaly'] == 0]
    X_train = normal_df[FEATURES]

    # Test on ALL data (normal + anomalies)
    X_test  = df[FEATURES]
    y_test  = df['is_anomaly']

    # ── 3. Scale features ─────────────────────────────────────
    # Brings all features to same scale so no single metric dominates
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 4. Train Isolation Forest ─────────────────────────────
    model = IsolationForest(
        n_estimators=100,      # number of trees
        contamination=0.10,    # expected % of anomalies (matches our 10%)
        random_state=42,
        max_samples='auto'
    )

    print("Training Isolation Forest...")
    model.fit(X_train_scaled)
    print("Training complete!")

    # ── 5. Save model and scaler ──────────────────────────────
    os.makedirs('model/artifacts', exist_ok=True)
    joblib.dump(model,  'model/artifacts/isolation_forest.pkl')
    joblib.dump(scaler, 'model/artifacts/scaler.pkl')
    print("Model saved to model/artifacts/")

    return model, scaler, X_test_scaled, y_test

if __name__ == '__main__':
    train()