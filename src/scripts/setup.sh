#!/bin/bash

# YouTube MLOps Pipeline Setup Script

echo "🚀 Setting up YouTube MLOps Pipeline..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p logs
mkdir -p metrics

# Copy environment template
if [ ! -f .env ]; then
    echo "🔧 Creating environment file..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
fi

# Initialize DVC
echo "🎯 Initializing DVC..."
if [