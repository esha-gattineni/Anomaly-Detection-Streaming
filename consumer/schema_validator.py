REQUIRED_FIELDS = [
    'timestamp',
    'host',
    'cpu_usage',
    'memory_usage',
    'latency_ms',
    'error_rate'
]

# Valid value ranges — anything outside = anomaly candidate or bad data
VALID_RANGES = {
    'cpu_usage':    (0, 100),
    'memory_usage': (0, 100),
    'latency_ms':   (0, 60000),
    'error_rate':   (0, 100)
}

def validate_event(event: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, reason)
    is_valid = True means event passed all checks
    """

    # Check 1: All required fields must exist
    for field in REQUIRED_FIELDS:
        if field not in event:
            return False, f"Missing field: {field}"

    # Check 2: host must be a non-empty string
    if not isinstance(event['host'], str) or len(event['host']) == 0:
        return False, "Invalid host value"

    # Check 3: Numeric fields must be within valid ranges
    for field, (min_val, max_val) in VALID_RANGES.items():
        val = event.get(field)
        if not isinstance(val, (int, float)):
            return False, f"Non-numeric value in {field}: {val}"
        if not (min_val <= val <= max_val):
            return False, f"Out of range {field}: {val}"

    return True, "OK"