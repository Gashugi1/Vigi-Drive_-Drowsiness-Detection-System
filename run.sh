#!/bin/bash
# VigiDrive Application Runner
# Production-ready startup script

echo "=== VigiDrive - Drowsiness Detection System ==="
echo "Starting application..."

# Navigate to src directory
cd "$(dirname "$0")/src"

# Check if virtual environment exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
else
    echo "WARNING: Virtual environment not found at ../venv/"
    echo "Please create a virtual environment: python3 -m venv venv"
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r ../requirements.txt
fi

# Create necessary directories
mkdir -p ../data/logs
mkdir -p ../data/instance
mkdir -p ../data/features

# Initialize database
echo "Initializing database..."
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('[OK] Database initialized')"

# Start application
echo ""
echo "Starting Flask application on http://localhost:5001"
echo "Press Ctrl+C to stop"
echo ""

python app.py
