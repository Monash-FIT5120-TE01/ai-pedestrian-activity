#!/bin/bash

set -e

# Use the project directory in EC2
cd /home/ubuntu/ai-pedestrian-activity

# Use virtual environment
PYTHON=".venv/bin/python"

# Log execution time
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting refresh weather pipeline"

# Step 1: Load live data
"$PYTHON" refresh_weather.py --output data/forecast --weather-provider auto


echo "[$(date '+%Y-%m-%d %H:%M:%S')] Refresh weather pipeline completed successfully"