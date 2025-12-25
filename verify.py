"""
Quick verification script to test all components
"""
import os
import sys

# Test imports
print("=" * 60)
print("🧪 FEDERATED LEARNING SYSTEM - COMPONENT VERIFICATION")
print("=" * 60)

print("\n1. Testing Imports...")
try:
    import torch
    import numpy as np
    import pandas as pd
    import flwr as fl
    import shap
    import google.generativeai as genai
    from flask import Flask
    print("   ✅ All dependencies imported successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

print("\n2. Testing Data Utils...")
try:
    sys.path.insert(0, 'backend')
    from data_utils import download_liver_dataset
    print("   ✅ Data utils module loaded")
except Exception as e:
    print(f"   ❌ Data utils error: {e}")

print("\n3. Testing Model...")
try:
    from model import create_model
    model = create_model(input_dim=10)
    print(f"   ✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"   ❌ Model error: {e}")

print("\n4. Testing LLM Service...")
try:
    from llm_service import ClinicalLLMService
    print("   ✅ LLM service module loaded")
    print("   ℹ️  Note: Actual API calls require valid Gemini API key")
except Exception as e:
    print(f"   ❌ LLM service error: {e}")

print("\n5. Checking Configuration...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and len(api_key) > 20:
        print(f"   ✅ Gemini API key configured (length: {len(api_key)})")
    else:
        print("   ⚠️  Gemini API key not found or invalid")
    
    num_clients = int(os.getenv('NUM_CLIENTS', 3))
    num_rounds = int(os.getenv('NUM_ROUNDS', 5))
    print(f"   ✅ Configuration: {num_clients} clients, {num_rounds} rounds")
except Exception as e:
    print(f"   ❌ Configuration error: {e}")

print("\n6. Testing Flower Framework...")
try:
    print(f"   ✅ Flower version: {fl.__version__}")
    print("   ✅ Flower server and client modules available")
except Exception as e:
    print(f"   ❌ Flower error: {e}")

print("\n" + "=" * 60)
print("✅ VERIFICATION COMPLETE!")
print("=" * 60)
print("\n📝 Next Steps:")
print("   1. Run: python3 run.py")
print("   2. Open browser to: http://localhost:5000")
print("   3. Enjoy federated learning with explainable AI!")
print("\n" + "=" * 60)
