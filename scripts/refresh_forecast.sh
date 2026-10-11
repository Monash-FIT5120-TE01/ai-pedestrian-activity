#!/bin/bash

set -e

# Use the project directory in EC2
cd /home/ubuntu/ai-pedestrian-activity

# Use virtual environment
PYTHON=".venv/bin/python"

# Log execution time
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting refresh pedestrian counts forecast pipeline"

# Load live data
"$PYTHON" predict.py --output forecast.json --fallback-seasonal

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refresh pedestrian counts forecast pipeline completed successfully"