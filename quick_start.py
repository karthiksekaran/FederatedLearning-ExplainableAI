#!/usr/bin/env python3
"""
Quick Start Script - Trains model and launches web interface
Skips federated learning for faster testing
"""
import subprocess
import sys
import time
import os

print("QUICK START - Federated Learning Demo")
print("=" * 60)
print()

# Step 1: Train model
print("Step 1/2: Training model locally...")
result = subprocess.run([sys.executable, 'simple_train.py'], capture_output=False)

if result.returncode != 0:
    print("\n❌ Training failed!")
    sys.exit(1)

print()
print("=" * 60)
print("Step 2/2: Starting web interface...")
print("=" * 60)

# Step 2: Start API server
port = os.getenv('FLASK_PORT', '5001')
print(f"\nStarting API server on port {port}...")
print(f"Open your browser to: http://localhost:{port}")
print("\nPress Ctrl+C to stop\n")

try:
    subprocess.run([sys.executable, 'backend/api_server.py'])
except KeyboardInterrupt:
    print("\n\nServer stopped. Goodbye!")
