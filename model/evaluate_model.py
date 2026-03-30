import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score
)

FEATURES = ['cpu_usage', 'memory_usage', 'latency_ms', 'error_rate']

def evaluate():
    # ── Load data + model ─────────────────────────────────────
    df     = pd.read_csv('data/telemetry_data.csv')
    model  = joblib.load('model/artifacts/isolation_forest.pkl')
    scaler = joblib.load('model/artifacts/scaler.pkl')

    X      = scaler.transform(df[FEATURES])
    y_true = df['is_anomaly'].values

    # ── Get predictions ───────────────────────────────────────
    # Convert Isolation Forest output: -1 → 1 (anomaly), 1 → 0 (normal)
    raw_pred = model.predict(X)
    y_pred   = np.where(raw_pred == -1, 1, 0)

    # ── Print metrics ─────────────────────────────────────────
    recall    = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print("=" * 45)
    print("       ANOMALY DETECTION RESULTS")
    print("=" * 45)
    print(classification_report(y_true, y_pred,
          target_names=['Normal', 'Anomaly']))
    print(f"Recall:    {recall:.2%}")
    print(f"Precision: {precision:.2%}")

    # ── MTTD comparison ───────────────────────────────────────
    # Threshold baseline: flags only when cpu > 90 OR latency > 800
    baseline_pred = ((df['cpu_usage'] > 90) |
                     (df['latency_ms'] > 800)).astype(int)

    baseline_recall = recall_score(y_true, baseline_pred)
    print(f"\nBaseline Recall (threshold): {baseline_recall:.2%}")
    print(f"Our Model Recall:             {recall:.2%}")
    print(f"Improvement:                  "
          f"{(recall - baseline_recall):.2%}")

    # ── Plot confusion matrix ─────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix — Isolation Forest')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model/artifacts/confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved to model/artifacts/")

    # ── Plot anomaly score distribution ──────────────────────
    scores = model.decision_function(X)
    df['anomaly_score'] = scores

    plt.figure(figsize=(8, 4))
    plt.hist(df[df['is_anomaly']==0]['anomaly_score'],
             bins=50, alpha=0.6, label='Normal', color='steelblue')
    plt.hist(df[df['is_anomaly']==1]['anomaly_score'],
             bins=50, alpha=0.6, label='Anomaly', color='tomato')
    plt.axvline(x=0, color='black', linestyle='--', label='Threshold (0)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model/artifacts/score_distribution.png', dpi=150)
    print("Score distribution saved to model/artifacts/")

if __name__ == '__main__':
    evaluate()