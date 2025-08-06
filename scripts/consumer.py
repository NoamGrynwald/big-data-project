#!/usr/bin/env python3

import json
import os
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_kafka(bootstrap_servers, max_retries=30, retry_delay=2):
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                consumer_timeout_ms=1000
            )
            consumer.close()
            logger.info("Kafka is available.")
            return True
        except NoBrokersAvailable:
            if attempt < 5:
                logger.info(f"Waiting for Kafka... (attempt {attempt + 1})")
            time.sleep(retry_delay)
        except Exception as e:
            logger.info(f"Kafka connection attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
    return False

def create_consumer(bootstrap_servers, topic, group_id):
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        max_poll_records=500,
        fetch_min_bytes=4096
    )

def main():
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    topic = os.getenv('KAFKA_TOPIC', 'fraud-alerts')
    group_id = os.getenv('KAFKA_GROUP_ID', 'fraud-consumer-group')

    logger.info("🚀 Starting Fraud Consumer...")
    logger.info(f"Bootstrap servers: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Group ID: {group_id}")

    if not wait_for_kafka(bootstrap_servers):
        logger.error("Could not connect to Kafka. Exiting.")
        return

    consumer = create_consumer(bootstrap_servers, topic, group_id)
    logger.info("✅ Successfully connected to Kafka and subscribed to fraud-alerts topic")

    try:
        count = 0
        for message in consumer:
            try:
                record = message.value
                amount = record.get("Amount", "N/A")
                score = record.get("fraud_score", "N/A")
                cls = record.get("Class", "?")
                timestamp = record.get("timestamp", "N/A")
                
                print(f"🚨 FRAUD ALERT:")
                print(f"   💰 Amount: ${amount}")
                print(f"   📊 Fraud Score: {score}")
                print(f"   🏷️  Class: {cls}")
                print(f"   ⏰ Timestamp: {timestamp}")
                print("-" * 50)
                
                count += 1
                if count % 10 == 0:
                    logger.info(f"Total fraud alerts processed: {count}")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        consumer.close()
        logger.info("Consumer closed.")

if __name__ == "__main__":
    main()