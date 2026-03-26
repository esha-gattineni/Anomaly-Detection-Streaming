import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

# ── Connect to Kafka ──────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC = 'telemetry-events'

# ── Normal ranges for each metric ────────────────────────────
NORMAL = {
    'cpu_usage':    (10, 75),
    'memory_usage': (20, 70),
    'latency_ms':   (50, 300),
    'error_rate':   (0.0, 2.0)
}

# ── Anomaly ranges (spikes) ───────────────────────────────────
ANOMALY = {
    'cpu_usage':    (90, 100),
    'memory_usage': (85, 99),
    'latency_ms':   (800, 3000),
    'error_rate':   (10.0, 50.0)
}

def generate_event(host, is_anomaly=False):
    """Build one telemetry event dict."""
    ranges = ANOMALY if is_anomaly else NORMAL
    return {
        'timestamp':    datetime.utcnow().isoformat(),
        'host':         host,
        'cpu_usage':    round(random.uniform(*ranges['cpu_usage']), 2),
        'memory_usage': round(random.uniform(*ranges['memory_usage']), 2),
        'latency_ms':   round(random.uniform(*ranges['latency_ms']), 2),
        'error_rate':   round(random.uniform(*ranges['error_rate']), 2),
        'is_anomaly':   is_anomaly          # label for later model training
    }

def run():
    # Simulate 5 different servers
    hosts = [fake.hostname() for _ in range(5)]
    print(f"Simulating hosts: {hosts}\n")
    print("Sending events to Kafka... Press Ctrl+C to stop.\n")

    event_count = 0

    while True:
        for host in hosts:
            # 10% chance of injecting an anomaly
            is_anomaly = random.random() < 0.10

            event = generate_event(host, is_anomaly)

            # Send to Kafka
            producer.send(TOPIC, value=event)

            event_count += 1
            label = "⚠ ANOMALY" if is_anomaly else "  normal "
            print(f"[{event_count}] {label} | {host} | "
                  f"CPU: {event['cpu_usage']}% | "
                  f"MEM: {event['memory_usage']}% | "
                  f"LAT: {event['latency_ms']}ms | "
                  f"ERR: {event['error_rate']}%")

        # Flush every batch of 5 events (one per host)
        producer.flush()
        time.sleep(0.5)   # send every 500ms

if __name__ == '__main__':
    run()