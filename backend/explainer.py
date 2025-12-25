"""
Explainability module using SHAP for model interpretation
"""
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from model import load_model


class ModelExplainer:
    """SHAP-based explainability for liver disease predictions"""
    
    def __init__(self, model, background_data, feature_names):
        """
        Initialize explainer
        
        Args:
            model: PyTorch model
            background_data: Background dataset for SHAP (numpy array)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        
        # Create model wrapper for SHAP
        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.FloatTensor(x)).numpy()
        
        self.predict_fn = model_predict
        
        # Initialize SHAP explainer (DeepExplainer for neural networks)
        background_tensor = torch.FloatTensor(background_data[:100])  # Use subset for efficiency
        self.explainer = shap.DeepExplainer(self.model, background_tensor)
    
    def explain_prediction(self, patient_data):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            patient_data: Single patient data (numpy array)
        
        Returns:
            Dictionary with SHAP values and feature importance
        """
        # Get SHAP values
        patient_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
        shap_values = self.explainer.shap_values(patient_tensor)
        
        # Extract SHAP values for the positive class
        if isinstance(shap_values, list):
            shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        
        # Create feature importance dictionary
        feature_importance = {
            name: float(val) for name, val in zip(self.feature_names, shap_vals)
        }
        
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'shap_values': shap_vals.tolist(),
            'feature_importance': feature_importance,
            'top_features': sorted_features[:5],
            'base_value': float(self.explainer.expected_value[0] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
        }
    
    def generate_waterfall_plot(self, patient_data, patient_values):
        """
        Generate SHAP waterfall plot
        
        Args:
            patient_data: Single patient data (numpy array)
            patient_values: Dictionary of {feature_name: value}
        
        Returns:
            Base64 encoded plot image
        """
        try:
            # Get SHAP values
            patient_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
            shap_values = self.explainer.shap_values(patient_tensor)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][0]
            else:
                shap_vals = shap_values[0]
            
            # Create explanation object
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
            
            # DeepExplainer returns (1, dim) for binary classification sometimes
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.reshape(-1)
                
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=patient_data,
                feature_names=self.feature_names
            )
            
            # Generate waterfall plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_base64
            
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_force_plot_data(self, patient_data):
        """
        Generate data for force plot visualization
        
        Args:
            patient_data: Single patient data (numpy array)
        
        Returns:
            Dictionary with force plot data
        """
        # Get SHAP values
        patient_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
        shap_values = self.explainer.shap_values(patient_tensor)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
        
        # Prepare data for frontend visualization
        features_data = []
        for i, (name, shap_val, data_val) in enumerate(zip(self.feature_names, shap_vals, patient_data)):
            features_data.append({
                'name': name,
                'value': float(data_val),
                'shap_value': float(shap_val),
                'effect': 'positive' if shap_val > 0 else 'negative'
            })
        
        # Sort by absolute SHAP value
        features_data.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'base_value': float(base_value),
            'features': features_data,
        }
    
    def generate_force_plot_image(self, patient_data):
        """
        Generate SHAP force plot as image
        """
        try:
            # Get SHAP values
            patient_tensor = torch.FloatTensor(patient_data).unsqueeze(0)
            shap_values = self.explainer.shap_values(patient_tensor)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[0][0]
            else:
                shap_vals = shap_values[0]
                
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.reshape(-1)

            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]

            plt.figure(figsize=(20, 3))
            shap.force_plot(
                base_value,
                shap_vals,
                patient_data,
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_base64
        except Exception as e:
            print(f"Error generating force plot: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_summary_plot(self):
        """
        Generate global summary plot
        """
        try:
            # We need background data + some test data ideally, but we only have background data stored in init usually.
            # But deep explainer uses background data. 
            # For summary plot we need a set of samples. 
            # We'll use the background data passed in init if available, or load test data.
            
            # Since self.explainer.shap_values requires a tensor, and we want to explain a batch.
            # We will load a small batch of test data for this summary.
            
            # Loading data inside here is a bit hacky but keeps it self-contained for now.
            from data_utils import load_test_data
            X_test, _ = load_test_data()
            if X_test is None:
                return None
            
            # Take a sample of 50-100 points
            X_sample = X_test[:50]
            X_tensor = torch.FloatTensor(X_sample)
            
            shap_output = self.explainer.shap_values(X_tensor)
            
            if isinstance(shap_output, list):
                shap_values = shap_output[0]
            else:
                shap_values = shap_output
                
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_base64
        except Exception as e:
            print(f"Error generating summary plot: {e}")
            return None


def create_explainer(model_path='models/global_model_final.pth', background_data=None, feature_names=None):
    """Factory function to create explainer"""
    model = load_model(model_path, input_dim=len(feature_names))
    return ModelExplainer(model, background_data, feature_names)


if __name__ == '__main__':
    # Test explainer
    print("🧪 Testing explainability module...")
    
    # This would normally use real data and model
    # For testing, we'll create dummy data
    from model import create_model
    import numpy as np
    
    model = create_model(input_dim=10)
    background_data = np.random.randn(100, 10)
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    explainer = ModelExplainer(model, background_data, feature_names)
    
    # Test single prediction explanation
    patient_data = np.random.randn(10)
    explanation = explainer.explain_prediction(patient_data)
    
    print(f"✅ SHAP values computed: {len(explanation['shap_values'])} features")
    print(f"✅ Top 5 features: {[f[0] for f in explanation['top_features']]}")
    
    # Test force plot data
    force_data = explainer.generate_force_plot_data(patient_data)
    print(f"✅ Force plot data generated: {len(force_data['features'])} features")
    
    print("\n✅ Explainability module test complete!")
