"""
LLM Service for Clinical Interpretation using Google Gemini
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class ClinicalLLMService:
    """Service for generating clinical interpretations using LLM"""
    
    def __init__(self, model_name='gemini-pro'):
        self.model = genai.GenerativeModel(model_name)
    
    def generate_clinical_interpretation(self, prediction, probability, feature_importance, patient_data):
        """
        Generate clinical interpretation for liver disease prediction
        
        Args:
            prediction: Binary prediction (0 or 1)
            probability: Prediction probability (0-1)
            feature_importance: Dictionary of feature names and their SHAP values
            patient_data: Dictionary of patient features
        
        Returns:
            Clinical interpretation text
        """
        
        # Create prompt
        diagnosis = "Liver Disease" if prediction == 1 else "No Liver Disease"
        confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
        
        # Format feature importance
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        features_text = "\n".join([f"- {name}: {value:.2f}" for name, value in top_features])
        
        # Format patient data
        patient_text = "\n".join([f"- {name}: {value}" for name, value in patient_data.items()])
        
        prompt = f"""You are a medical AI assistant helping doctors interpret liver disease predictions.

**Prediction Results:**
- Diagnosis: {diagnosis}
- Confidence: {confidence:.1f}%

**Patient Data:**
{patient_text}

**Most Influential Features (SHAP values):**
{features_text}

Please provide a concise clinical interpretation in the following format:

1. **Patient Status**: Brief assessment of the patient's condition
2. **Diagnosis Interpretation**: Explain the prediction in medical terms
3. **Key Factors**: Describe the most important features that influenced this prediction
4. **Possible Treatments**: Suggest 2-3 evidence-based treatment options or next steps

Keep the response professional, concise, and actionable for healthcare providers. Use medical terminology where appropriate but ensure clarity."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating clinical interpretation: {str(e)}\n\nFallback: The model predicts {diagnosis} with {confidence:.1f}% confidence. Top contributing factors: {', '.join([name for name, _ in top_features[:3]])}."
    
    def summarize_federated_results(self, round_num, global_accuracy, client_metrics):
        """Generate summary of federated learning round"""
        
        client_info = "\n".join([
            f"- Hospital {m['client_id']}: Train Acc={m['train_accuracy']:.2%}, Val Acc={m['val_accuracy']:.2%}"
            for m in client_metrics
        ])
        
        prompt = f"""You are an AI assistant summarizing federated learning progress for medical data.

**Training Round {round_num} Results:**
- Global Model Accuracy: {global_accuracy:.2%}

**Client Performance:**
{client_info}

Provide a brief 2-3 sentence summary of:
1. Overall training progress
2. Model convergence status
3. Any notable observations about client performance variations

Keep it concise and technical."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Round {round_num}: Global accuracy {global_accuracy:.2%}. Training progressing across {len(client_metrics)} healthcare centers."


def test_llm_service():
    """Test LLM service"""
    service = ClinicalLLMService()
    
    # Test prediction
    sample_prediction = 1
    sample_probability = 0.85
    sample_features = {
        'Age': 0.45,
        'Total_Bilirubin': 0.32,
        'Alkaline_Phosphotase': -0.28,
        'Albumin': -0.22,
        'Albumin_Globulin_Ratio': -0.18
    }
    sample_patient = {
        'Age': 65,
        'Total_Bilirubin': 2.5,
        'Alkaline_Phosphotase': 320,
        'Albumin': 3.2,
        'Albumin_Globulin_Ratio': 0.95
    }
    
    print("🤖 Testing LLM Clinical Interpretation...\n")
    interpretation = service.generate_clinical_interpretation(
        sample_prediction, sample_probability, sample_features, sample_patient
    )
    print(interpretation)
    
    print("\n" + "="*50 + "\n")
    
    # Test federated summary
    sample_client_metrics = [
        {'client_id': 1, 'train_accuracy': 0.82, 'val_accuracy': 0.79},
        {'client_id': 2, 'train_accuracy': 0.85, 'val_accuracy': 0.83},
        {'client_id': 3, 'train_accuracy': 0.79, 'val_accuracy': 0.77}
    ]
    
    print("🤖 Testing Federated Learning Summary...\n")
    summary = service.summarize_federated_results(3, 0.81, sample_client_metrics)
    print(summary)


if __name__ == '__main__':
    test_llm_service()
