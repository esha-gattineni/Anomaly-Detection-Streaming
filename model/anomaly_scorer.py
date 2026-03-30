import joblib
import numpy as np
import pandas as pd

FEATURES = ['cpu_usage', 'memory_usage', 'latency_ms', 'error_rate']

# Load model and scaler once at startup
model  = joblib.load('model/artifacts/isolation_forest.pkl')
scaler = joblib.load('model/artifacts/scaler.pkl')

def score_event(event: dict) -> dict:
    """
    Takes one telemetry event dict.
    Returns the event enriched with anomaly_score and is_flagged.
    """
    # Extract feature values in correct order
    features = [[
        event['cpu_usage'],
        event['memory_usage'],
        event['latency_ms'],
        event['error_rate']
    ]]

    # Scale features
    scaled = scaler.transform(features)

    # Isolation Forest score:
    # More negative = more anomalous
    # Positive = normal
    raw_score = model.decision_function(scaled)[0]

    # Predict: -1 = anomaly, 1 = normal
    prediction = model.predict(scaled)[0]

    event['anomaly_score'] = round(float(raw_score), 4)
    event['is_flagged']    = bool(prediction == -1)

    return event


def score_batch(events: list[dict]) -> list[dict]:
    """Score a list of events at once — more efficient."""
    df = pd.DataFrame(events)[FEATURES]
    scaled = scaler.transform(df)

    scores      = model.decision_function(scaled)
    predictions = model.predict(scaled)

    for i, event in enumerate(events):
        event['anomaly_score'] = round(float(scores[i]), 4)
        event['is_flagged']    = bool(predictions[i] == -1)

    return events