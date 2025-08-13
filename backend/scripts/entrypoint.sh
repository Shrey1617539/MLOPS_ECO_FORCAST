#!/bin/bash
set -e

# Wait for any dependent services if needed
# Example: wait-for-it.sh database:5432 -t 60

# Run migrations or setup scripts if needed
# python -m src.data.storage_setup

# Start the application
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
