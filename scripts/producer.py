#!/usr/bin/env python3

import json
import time
import os
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_kafka(bootstrap_servers, max_retries=30, retry_delay=2):
    """Wait for Kafka to be available"""
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            producer.close()
            logger.info("Kafka is available!")
            return True
        except NoBrokersAvailable:
            logger.info(f"Kafka not available yet, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        except Exception as e:
            logger.info(f"Kafka connection attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
    
    logger.error("Could not connect to Kafka after maximum retries")
    return False

def create_producer(bootstrap_servers):
    """Create Kafka producer"""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
        batch_size=16384,
        linger_ms=10,
        buffer_memory=33554432
    )

def read_and_send_json(file_path, producer, topic):
    """Read JSON file and send each line to Kafka"""
    logger.info(f"Starting to read file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_count = 0
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        # Parse JSON to validate it
                        json_data = json.loads(line)
                        
                        # Send to Kafka
                        future = producer.send(topic, json_data)
                        
                        line_count += 1
                        if line_count % 100 == 0:
                            logger.info(f"Sent {line_count} records to Kafka")
                        
                        # Small delay to simulate real-time streaming
                        time.sleep(0.015)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {line_count + 1}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_count + 1}: {e}")
                        continue
            
            # Ensure all messages are sent
            producer.flush()
            logger.info(f"Finished sending {line_count} records to Kafka topic '{topic}'")
            
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise

def main():
    # Configuration
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic = os.getenv('KAFKA_TOPIC', 'creditcard-data')
    json_file_path = '/app/data/creditcard_json.json'  # Updated path for your structure
    
    logger.info("Starting JSON Producer")
    logger.info(f"Kafka servers: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"JSON file: {json_file_path}")
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        logger.error(f"Data file not found: {json_file_path}")
        logger.info("Available files in /app/data:")
        try:
            for f in os.listdir('/app/data'):
                logger.info(f"  - {f}")
        except:
            logger.info("  - Could not list directory")
        return
    
    # Wait for Kafka to be available
    if not wait_for_kafka(bootstrap_servers):
        return
    
    # Create producer
    producer = create_producer(bootstrap_servers)
    
    try:
        # Send data
        read_and_send_json(json_file_path, producer, topic)
        
        # Send end-of-stream marker
        end_marker = {"__END_OF_STREAM__": True}
        producer.send(topic, end_marker)
        producer.flush()
        logger.info("Sent end-of-stream marker")
        
    except KeyboardInterrupt:
        logger.info("Producer interrupted by user")
    except Exception as e:
        logger.error(f"Producer error: {e}")
        raise
    finally:
        producer.close()
        logger.info("Producer closed")

if __name__ == "__main__":
    main()