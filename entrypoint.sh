#!/bin/sh

echo "Starting container..."

echo "Running dvc pull..."
dvc pull

echo "Starting API..."

exec uvicorn mlops_stock_project.api.app:app \
  --host 0.0.0.0 \
  --port ${PORT:-8080}