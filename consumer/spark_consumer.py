import json
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, BooleanType
)
from schema_validator import validate_event

# ── 1. Define the expected schema of each event ───────────────
EVENT_SCHEMA = StructType([
    StructField("timestamp",    StringType(),  True),
    StructField("host",         StringType(),  True),
    StructField("cpu_usage",    FloatType(),   True),
    StructField("memory_usage", FloatType(),   True),
    StructField("latency_ms",   FloatType(),   True),
    StructField("error_rate",   FloatType(),   True),
    StructField("is_anomaly",   BooleanType(), True),
])

# ── 2. Create Spark Session ───────────────────────────────────
spark = SparkSession.builder \
    .appName("TelemetryAnomalyDetection") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0"
    ) \
    .config(
        # Exclude the jar that fails to download from Maven
        "spark.jars.excludes",
        "com.google.code.findbugs:jsr305"
    ) \
    .config(
        "spark.streaming.kafka.maxRatePerPartition", "1000"
    ) \
    .config(
        "spark.sql.streaming.kafka.maxOffsetsPerTrigger", "5000"
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")  # reduces noisy logs

# ── 3. Read stream from Kafka ─────────────────────────────────
raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "telemetry-events") \
    .option("startingOffsets", "latest") \
    .load()

# ── 4. Decode bytes → JSON string → structured columns ────────
parsed_stream = raw_stream \
    .selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), EVENT_SCHEMA).alias("data")) \
    .select("data.*")

# ── 5. Schema Validation via UDF ─────────────────────────────
# UDF = User Defined Function: runs your Python function on each row

def is_valid_event(timestamp, host, cpu, memory, latency, error):
    """Wraps validate_event for use inside Spark."""
    event = {
        'timestamp':    timestamp,
        'host':         host,
        'cpu_usage':    cpu,
        'memory_usage': memory,
        'latency_ms':   latency,
        'error_rate':   error
    }
    valid, reason = validate_event(event)
    return valid

validate_udf = udf(is_valid_event, BooleanType())

validated_stream = parsed_stream.withColumn(
    "is_valid",
    validate_udf(
        col("timestamp"),
        col("host"),
        col("cpu_usage"),
        col("memory_usage"),
        col("latency_ms"),
        col("error_rate")
    )
)

# ── 6. Split into clean and quarantine streams ────────────────
clean_stream = validated_stream.filter(col("is_valid") == True)
bad_stream   = validated_stream.filter(col("is_valid") == False)

# ── 7. Output — print clean events to console ─────────────────
clean_query = clean_stream.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .trigger(processingTime="5 seconds") \
    .start()

# ── 8. Output — save bad events to a quarantine folder ────────
os.makedirs("../quarantine", exist_ok=True)

bad_query = bad_stream.writeStream \
    .outputMode("append") \
    .format("json") \
    .option("path", "../quarantine") \
    .option("checkpointLocation", "../checkpoints/bad") \
    .trigger(processingTime="5 seconds") \
    .start()

print("✅ Spark Streaming started. Listening to Kafka...")
print("   Clean events → console")
print("   Bad events   → ../quarantine/")
print("   Press Ctrl+C to stop.\n")

# Keep the job running until manually stopped
spark.streams.awaitAnyTermination()