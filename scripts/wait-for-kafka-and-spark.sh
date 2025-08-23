#!/bin/bash
set -e

echo "🔁 Waiting for Kafka on kafka:29092..."
timeout=60
elapsed=0
while ! nc -z kafka 29092; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [ $elapsed -ge $timeout ]; then
    echo "❌ Timeout waiting for Kafka"
    exit 1
  fi
done
echo "✅ Kafka is up!"

# Wait a bit more for Kafka to be fully ready
sleep 5

# Now run the producer
echo "🚀 Starting producer..."
exec python scripts/producer.py