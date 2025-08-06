#!/usr/bin/env python3

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.streaming import DataStreamWriter

def create_spark_session():
    return SparkSession.builder \
        .appName("KafkaFraudDetection") \
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def define_transaction_schema():
    """Schema matching your actual credit card data"""
    return StructType([
        StructField("Amount", DoubleType(), True),
        StructField("Class", StringType(), True),
        StructField("Time", DoubleType(), True),
        StructField("V1", DoubleType(), True),
        StructField("V2", DoubleType(), True),
        StructField("V3", DoubleType(), True),
        StructField("V4", DoubleType(), True),
        StructField("V5", DoubleType(), True),
        StructField("V6", DoubleType(), True),
        StructField("V7", DoubleType(), True),
        StructField("V8", DoubleType(), True),
        StructField("V9", DoubleType(), True),
        StructField("V10", DoubleType(), True),
        StructField("V11", DoubleType(), True),
        StructField("V12", DoubleType(), True),
        StructField("V13", DoubleType(), True),
        StructField("V14", DoubleType(), True),
        StructField("V15", DoubleType(), True),
        StructField("V16", DoubleType(), True),
        StructField("V17", DoubleType(), True),
        StructField("V18", DoubleType(), True),
        StructField("V19", DoubleType(), True),
        StructField("V20", DoubleType(), True),
        StructField("V21", DoubleType(), True),
        StructField("V22", DoubleType(), True),
        StructField("V23", DoubleType(), True),
        StructField("V24", DoubleType(), True),
        StructField("V25", DoubleType(), True),
        StructField("V26", DoubleType(), True),
        StructField("V27", DoubleType(), True),
        StructField("V28", DoubleType(), True)
    ])

def detect_fraud(df):
    """Enhanced fraud detection using multiple features"""
    return df.withColumn("clean_class", 
                        regexp_replace(col("Class"), "'", "")) \
        .withColumn("fraud_score",
                when(col("Amount") > 5000, 0.95)
                .when(col("Amount") > 2000, 0.8)
                .when(col("Amount") > 1000, 0.6)
                .when(col("Amount") > 500, 0.4)
                .when(col("Amount") > 200, 0.3)
                .when((abs(col("V1")) > 3) | (abs(col("V2")) > 3), 0.7)
                .otherwise(0.1)) \
        .withColumn("is_fraud", col("fraud_score") > 0.5)

def process_stream(spark):
    print("🚀 Starting Fraud Detection Stream Processing...")
    
    schema = define_transaction_schema()

    # Read from Kafka topic: creditcard-data
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("subscribe", "creditcard-data") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    print("✅ Connected to Kafka stream")

    # Parse JSON from Kafka messages
    parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json("json_str", schema).alias("data")) \
        .select("data.*") \
        .na.drop(subset=["Amount"])

    print("✅ Parsing JSON messages")

    # Apply fraud detection with timestamp
    fraud_df = detect_fraud(parsed_df).withColumn("timestamp", current_timestamp())

    # Filter only frauds
    fraud_alerts_df = fraud_df.filter(col("is_fraud") == True)

    # Convert to Kafka-compatible key-value JSON
    kafka_ready_df = fraud_alerts_df.selectExpr(
        "CAST(Amount AS STRING) AS key",
        """to_json(named_struct(
            'Amount', Amount,
            'Class', clean_class,
            'fraud_score', fraud_score,
            'timestamp', CAST(timestamp AS STRING),
            'V1', V1,
            'V2', V2,
            'Time', Time
        )) AS value"""
    )

    print("✅ Setting up output streams")

    # Send to fraud-alerts topic
    kafka_query = kafka_ready_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:29092") \
        .option("topic", "fraud-alerts") \
        .option("checkpointLocation", "/tmp/checkpoints/fraud-alerts") \
        .outputMode("append") \
        .trigger(processingTime='5 seconds') \
        .start()

    # Console output for debugging
    console_query = fraud_alerts_df.select("Amount", "clean_class", "fraud_score", "timestamp") \
        .writeStream \
        .format("console") \
        .outputMode("append") \
        .trigger(processingTime='5 seconds') \
        .option("truncate", "false") \
        .start()

    print("✅ Fraud detection pipeline started!")
    print("📊 Monitoring for fraudulent transactions...")

    try:
        kafka_query.awaitTermination()
    except KeyboardInterrupt:
        print("🛑 Stopping fraud detection...")
        kafka_query.stop()
        console_query.stop()

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")  # Reduce Spark logs
    
    print("🔥 Spark Session Created Successfully")
    print("⏳ Waiting for Kafka messages...")
    
    process_stream(spark)

if __name__ == "__main__":
    main()