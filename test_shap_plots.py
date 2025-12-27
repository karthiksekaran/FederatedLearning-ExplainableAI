import sys
import os
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend

sys.path.insert(0, os.path.abspath('backend'))
from model import load_model
from explainer import create_explainer
from data_utils import load_test_data

def test():
    print("Testing SHAP plot generation...")
    
    if not os.path.exists('models/global_model_final.pth'):
        print("Model not found")
        return

    with open('data/liver_data.pkl', 'rb') as f:
        cached_data = pickle.load(f)
        feature_names = cached_data['feature_names']
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    X_test, y_test = load_test_data()
    
    explainer = create_explainer(
        'models/global_model_final.pth',
        X_test[:20], # Small background
        feature_names
    )
    
    # Mock patient data
    patient_idx = 0
    patient_data = X_test[patient_idx]
    
    print("Generating Force Plot... ", end="")
    force_img = explainer.generate_force_plot_image(patient_data)
    if force_img and len(force_img) > 100:
        print("✅ Success (size: {} bytes)".format(len(force_img)))
    else:
        print("❌ Failed")
        
    print("Generating Summary Plot... ", end="")
    summary_img = explainer.generate_summary_plot()
    if summary_img and len(summary_img) > 100:
        print("✅ Success (size: {} bytes)".format(len(summary_img)))
    else:
        print("❌ Failed")

if __name__ == '__main__':
    test()
