#!/bin/bash

echo "🧬 DNA Sequence Analyzer - Starting Server"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d "../venv" ]; then
    echo "✓ Activating virtual environment..."
    source ../venv/bin/activate
fi

# Run migrations
echo "✓ Running migrations..."
python manage.py migrate

# Create media directory
echo "✓ Creating media directory..."
mkdir -p media

echo ""
echo "=========================================="
echo "🚀 Starting Django development server..."
echo "📱 Access the app at: http://127.0.0.1:8000/"
echo "=========================================="
echo ""

# Start server
python manage.py runserver
