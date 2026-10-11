#!/bin/bash

set -e

# Use the project directory in EC2
cd /home/ubuntu/ai-pedestrian-activity

# Use virtual environment
PYTHON=".venv/bin/python"

# Log execution time
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting refresh live pedestrian counts pipeline"

# Step 1: Load live data
"$PYTHON" load_data.py --only-live

# Step 2: Clean live data (runs only if Step 1 succeeds)
"$PYTHON" clean_data.py --live data/raw/live_snapshot.csv --only-live

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refresh live pedestrian counts pipeline completed successfully"