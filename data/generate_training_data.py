import pandas as pd
import numpy as np
import random

def generate_dataset(n_normal=5000, n_anomaly=500, save=True):
    """
    Generate labeled telemetry dataset.
    90% normal events, 10% anomalies — matches real production ratio.
    """
    random.seed(42)
    np.random.seed(42)

    # ── Normal events ─────────────────────────────────────────
    normal = pd.DataFrame({
    'cpu_usage':    np.random.uniform(10, 85,  n_normal),   # overlaps anomaly
    'memory_usage': np.random.uniform(20, 80,  n_normal),   # overlaps anomaly
    'latency_ms':   np.random.uniform(50, 500, n_normal),   # overlaps anomaly
    'error_rate':   np.random.uniform(0,  5,   n_normal),   # overlaps anomaly
    'is_anomaly':   [0] * n_normal
})

    # ── Anomaly events (spikes) ───────────────────────────────
    anomaly = pd.DataFrame({
    'cpu_usage':    np.random.uniform(70, 100,  n_anomaly),  # overlaps normal
    'memory_usage': np.random.uniform(60, 99,   n_anomaly),  # overlaps normal
    'latency_ms':   np.random.uniform(300, 3000,n_anomaly),  # overlaps normal
    'error_rate':   np.random.uniform(3,  50,   n_anomaly),  # overlaps normal
    'is_anomaly':   [1] * n_anomaly
})

    # ── Combine and shuffle ───────────────────────────────────
    df = pd.concat([normal, anomaly]).sample(frac=1).reset_index(drop=True)

    if save:
        df.to_csv('data/telemetry_data.csv', index=False)
        print(f"Dataset saved: {len(df)} events "
              f"({n_normal} normal, {n_anomaly} anomaly)")

    return df

if __name__ == '__main__':
    generate_dataset()