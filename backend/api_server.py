"""
Flask API Server for Federated Learning Web Interface
Provides endpoints for training monitoring and predictions
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import pickle
import os
import json
from datetime import datetime

from model import load_model, create_model
from explainer import create_explainer
from llm_service import ClinicalLLMService
from data_utils import get_data_loaders

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Global variables
llm_service = ClinicalLLMService()
explainer = None
model = None
feature_names = None
scaler = None

# Training state
training_state = {
    'is_training': False,
    'current_round': 0,
    'total_rounds': 0,
    'history': []
}


@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current training status"""
    # Try to load from file if available
    if os.path.exists('data/training_state.json'):
        try:
            with open('data/training_state.json', 'r') as f:
                loaded_state = json.load(f)
                training_state.update(loaded_state)
        except Exception as e:
            print(f"Could not load training state: {e}")
    return jsonify(training_state)


@app.route('/api/training/history', methods=['GET'])
def get_training_history():
    """Get training history"""
    return jsonify(training_state['history'])


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with explanation"""
    global model, explainer, feature_names, scaler
    
    try:
        # Get patient data
        data = request.json
        patient_values = data.get('patient_data', {})
        
        # Load model if not loaded
        if model is None:
            if not os.path.exists('models/global_model_final.pth'):
                return jsonify({'error': 'Model not trained yet. Please run training first.'}), 400
            
            # Load feature names and scaler
            with open('data/liver_data.pkl', 'rb') as f:
                cached_data = pickle.load(f)
                feature_names = cached_data['feature_names']
            
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            
            model = load_model('models/global_model_final.pth', input_dim=len(feature_names))
            
            # Load background data for explainer
            X_test, y_test = __import__('data_utils').load_test_data()
            explainer = create_explainer(
                'models/global_model_final.pth',
                X_test[:100],
                feature_names
            )
        
        # Convert patient values to array
        patient_array = np.array([patient_values.get(name, 0) for name in feature_names])
        patient_array = scaler.transform(patient_array.reshape(1, -1))[0]
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prob = model(torch.FloatTensor(patient_array).unsqueeze(0))
            probability = prob.item()
            prediction = 1 if probability > 0.5 else 0
        
        # Get explanation
        explanation = explainer.explain_prediction(patient_array)
        force_plot_data = explainer.generate_force_plot_data(patient_array)
        
        # Generate clinical interpretation
        clinical_interpretation = llm_service.generate_clinical_interpretation(
            prediction=prediction,
            probability=probability,
            feature_importance=explanation['feature_importance'],
            patient_data=patient_values
        )
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'diagnosis': 'Liver Disease' if prediction == 1 else 'No Liver Disease',
            'confidence': float(probability * 100 if prediction == 1 else (1 - probability) * 100),
            'explanation': explanation,
            'force_plot': force_plot_data,
            'clinical_interpretation': clinical_interpretation
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if os.path.exists('models/global_model_final.pth'):
            model_path = 'models/global_model_final.pth'
            size = os.path.getsize(model_path)
            modified = datetime.fromtimestamp(os.path.getmtime(model_path))
            
            return jsonify({
                'exists': True,
                'size': size,
                'last_modified': modified.isoformat(),
                'path': model_path
            })
        else:
            return jsonify({'exists': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get model and training configuration"""
    try:
        from model import LiverDiseaseClassifier
        import torch
        
        # Get environment config
        config = {
            'model': {
                'name': 'LiverDiseaseClassifier',
                'type': 'Neural Network',
                'framework': 'PyTorch',
                'architecture': {
                    'input_dim': 10,
                    'hidden_layers': [64, 32],
                    'output_dim': 1,
                    'activation': 'ReLU',
                    'output_activation': 'Sigmoid',
                    'dropout_rate': 0.3
                }
            },
            'training': {
                'num_clients': int(os.getenv('NUM_CLIENTS', 3)),
                'num_rounds': int(os.getenv('NUM_ROUNDS', 5)),
                'batch_size': int(os.getenv('BATCH_SIZE', 32)),
                'learning_rate': float(os.getenv('LEARNING_RATE', 0.001)),
                'epochs_per_round': int(os.getenv('EPOCHS_PER_ROUND', 5))
            },
            'dataset': {
                'name': 'UCI Liver Patient Dataset (ILPD)',
                'task': 'Binary Classification',
                'features': 10,
                'target': 'Liver Disease Presence'
            }
        }
        
        # Add parameter count if model exists
        if os.path.exists('models/global_model_final.pth'):
            try:
                temp_model = LiverDiseaseClassifier(input_dim=10)
                total_params = sum(p.numel() for p in temp_model.parameters())
                trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
                config['model']['total_parameters'] = total_params
                config['model']['trainable_parameters'] = trainable_params
            except Exception as e:
                print(f"Could not load model for param count: {e}")
        
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/features', methods=['GET'])
def get_features():
    """Get feature names and sample ranges"""
    try:
        if os.path.exists('data/liver_data.pkl'):
            with open('data/liver_data.pkl', 'rb') as f:
                data = pickle.load(f)
                X = data['X']
                feature_names_list = data['feature_names']
            
            # Calculate feature statistics
            features_info = []
            for i, name in enumerate(feature_names_list):
                features_info.append({
                    'name': name,
                    'min': round(float(np.min(X[:, i])), 3),
                    'max': round(float(np.max(X[:, i])), 3),
                    'mean': round(float(np.mean(X[:, i])), 3),
                    'std': round(float(np.std(X[:, i])), 3)
                })
            
            return jsonify({'features': features_info})
        else:
            return jsonify({'error': 'Data not loaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Trigger federated training (this would actually be handled by run.py)"""
    return jsonify({
        'message': 'Training should be started via run.py script',
        'command': 'python run.py'
    })


def update_training_state(round_num, metrics):
    """Update training state (called from external process)"""
    training_state['current_round'] = round_num
    training_state['history'].append({
        'round': round_num,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    })
    
    # Save to file
    with open('data/training_state.json', 'w') as f:
        json.dump(training_state, f)


if __name__ == '__main__':
    PORT = int(os.getenv('FLASK_PORT', 5000))
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    
    print(f"Starting Flask API server at http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)
