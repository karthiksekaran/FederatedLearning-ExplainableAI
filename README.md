# FedMedAI - Federated Learning for Healthcare

Privacy-preserving federated learning system for medical data analysis using PyTorch and Flower framework.

## Overview

FedMedAI implements federated learning to train machine learning models across multiple healthcare institutions without sharing sensitive patient data. The system includes explainable AI capabilities using SHAP and natural language interpretations via Google Gemini.

## Features

- **Federated Learning**: Train models collaboratively across multiple institutions using Flower framework
- **Privacy-Preserving**: Raw patient data never leaves local institutions
- **Explainable AI**: SHAP (SHapley Additive exPlanations) for model interpretability
- **Clinical Insights**: LLM-powered natural language interpretations via Google Gemini
- **Binary Classification**: Liver disease prediction using UCI ILPD dataset
- **Web Interface**: Real-time training monitoring and prediction interface

## Technology Stack

- **Framework**: Flower (Federated Learning)
- **ML Library**: PyTorch
- **Explainability**: SHAP
- **LLM**: Google Gemini API
- **Backend**: Flask
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Dataset**: UCI Liver Patient Dataset (ILPD)

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FederatedLearning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5001
FLOWER_SERVER_PORT=8080

# Training Configuration
NUM_CLIENTS=3
NUM_ROUNDS=5
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS_PER_ROUND=5

# Data Configuration
TRAIN_TEST_SPLIT=0.8
RANDOM_SEED=42
```

## Quick Start

### Simple Training Mode

For quick testing without federated learning:

```bash
python3 quick_start.py
```

This will:
1. Download and prepare the UCI Liver dataset
2. Train the model locally (20 epochs)
3. Start the web interface at http://localhost:5001

### Full Federated Learning Mode

For complete federated learning with multiple clients:

```bash
python3 run.py
```

This will:
1. Download and prepare data
2. Split data across 3 simulated hospitals
3. Start Flower server
4. Launch 3 client nodes
5. Train for 5 federated rounds
6. Start the web interface

## Usage Guide

### Dashboard View

After starting the server, navigate to http://localhost:5001

The dashboard displays:
- **Model Configuration**: Shows architecture (10→64→32→1), parameter count, and training setup
- **Training Progress**: Real-time updates every 3 seconds showing current round and metrics
- **System Status**: Green dot indicates training complete, amber (pulsing) indicates training in progress

### Making Predictions

1. Click the **Prediction** tab
2. Enter patient data in the form:
   - **Age**: Integer value (e.g., 45)
   - **Gender**: 0 for Female, 1 for Male
   - **Biochemical Markers**: Decimal values up to 2 decimal places (e.g., 1.25, 250.50)
3. Click **Analyze Patient Data**
4. View results:
   - **Diagnosis**: Liver Disease or No Liver Disease
   - **Confidence**: Percentage likelihood
   - **Feature Importance**: SHAP values showing which factors influenced the prediction
   - **Clinical Interpretation**: Natural language explanation from LLM

### Input Field Details

| Field | Type | Example | Range |
|-------|------|---------|-------|
| Age | Integer | 45 | 4-90 |
| Gender | Integer | 1 | 0 (Female) or 1 (Male) |
| Total Bilirubin | Decimal (2 places) | 1.20 | 0.4-75.0 |
| Direct Bilirubin | Decimal (2 places) | 0.40 | 0.1-19.7 |
| Alkaline Phosphatase | Decimal (2 places) | 250.00 | 63-2110 |
| Alamine Aminotransferase | Decimal (2 places) | 35.50 | 10-2000 |
| Aspartate Aminotransferase | Decimal (2 places) | 60.75 | 10-4929 |
| Total Proteins | Decimal (2 places) | 6.50 | 2.7-9.6 |
| Albumin | Decimal (2 places) | 3.50 | 0.9-5.5 |
| Albumin/Globulin Ratio | Decimal (2 places) | 1.10 | 0.3-2.8 |

## Project Structure

```
FederatedLearning/
├── backend/
│   ├── api_server.py             # Flask API server
│   ├── data_utils.py            # Data download and preprocessing
│   ├── model.py                 # PyTorch model architecture
│   ├── federated_server.py      # Flower server implementation
│   ├── federated_client.py      # Flower client implementation
│   ├── explainer.py             # SHAP explainability
│   └── llm_service.py           # Google Gemini integration
├── frontend/
│   ├── index.html               # Web interface
│   └── app.js                   # Frontend logic
├── data/                        # Generated datasets and training state
├── models/                      # Trained model files
├── .env                         # Environment configuration
├── requirements.txt             # Python dependencies
├── quick_start.py               # Quick start script (simple training)
├── run.py                       # Full federated learning pipeline
└── README.md                    # This file
```

## Model Architecture

```
Input (10 features) → Dense(64) → BatchNorm → ReLU → Dropout(0.3)
                   → Dense(32) → BatchNorm → ReLU → Dropout(0.3)
                   → Dense(1) → Sigmoid → Output (probability)
```

**Total Parameters**: Approximately 3,000 trainable parameters

## Dataset

**UCI Liver Patient Dataset (ILPD)**
- Task: Binary classification (liver disease vs. healthy)
- Samples: ~583 patients
- Features:
  - Age
  - Gender
  - Total Bilirubin
  - Direct Bilirubin
  - Alkaline Phosphatase
  - Alamine Aminotransferase
  - Aspartate Aminotransferase
  - Total Proteins
  - Albumin
  - Albumin/Globulin Ratio

## Training

### Local Training (Simple Mode)

```bash
python3 simple_train.py
```

- Trains for 20 epochs
- Saves model to `models/global_model_final.pth`
- Writes training state to `data/training_state.json`
- Frontend polls this file for real-time updates

### Federated Training (Full Mode)

The system simulates 3 hospitals with federated learning:

1. Data is split across 3 clients
2. Each client trains locally on their data
3. Only model weights are shared with central server
4. Server aggregates weights using FedAvg algorithm
5. Process repeats for 5 rounds

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Model and training configuration |
| `/api/status` | GET | Current training status and history |
| `/api/model/info` | GET | Model file information |
| `/api/data/features` | GET | Feature names and statistics |
| `/api/predict` | POST | Make prediction with SHAP explanation |

### Example Prediction Request

```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "Age": 45,
      "Total_Bilirubin": 1.2,
      "Direct_Bilirubin": 0.4,
      "Alkaline_Phosphatase": 250,
      "Alamine_Aminotransferase": 35,
      "Aspartate_Aminotransferase": 60,
      "Total_Proteins": 6.5,
      "Albumin": 3.5,
      "Albumin_Globulin_Ratio": 1.1,
      "Gender": 1
    }
  }'
```

## Development

### Running Individual Components

**Test Model**:
```bash
python backend/model.py
```

**Test Data Utils**:
```bash
python backend/data_utils.py
```

**Test Explainer**:
```bash
python backend/explainer.py
```

**Test LLM Service**:
```bash
python backend/llm_service.py
```

### Running API Server Only

```bash
python backend/api_server.py
```

## Troubleshooting

### Port Already in Use

If port 5001 is in use, change `FLASK_PORT` in `.env` file:

```env
FLASK_PORT=5002
```

### API Key Issues

Ensure your `.env` file contains a valid Gemini API key:
- Get a key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Add to `.env` as `GEMINI_API_KEY=your_key_here`

### Training Not Updating

- Ensure `data/training_state.json` is being created
- Check console output for errors
- Verify training actually started (check console logs)

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Input Validation Errors

- Age must be an integer (no decimal points)
- Gender must be 0 or 1
- All other fields accept decimal values with up to 2 decimal places

## Cross-Platform Notes

### Windows

- Use `python` instead of `python3`
- Use backslashes for paths or forward slashes
- Ensure Python is added to PATH

### macOS

- Port 5000 may be used by AirPlay Receiver
- Use port 5001 (default) or change in `.env`

### Linux

- Install Python 3.8+ via package manager
- May need `python3-pip` separately

## Privacy and Security

- Raw patient data never leaves local clients
- Only model weights are shared with server
- API keys stored in `.env` (gitignored)
- HTTPS recommended for production deployment

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Flower Team - Federated learning framework
- UCI ML Repository - ILPD dataset
- Google - Gemini API
- SHAP - Model explainability library

## Support

For issues, questions, or contributions:
1. Check existing issues in the repository
2. Review code documentation
3. Test individual components using commands above

---

**Built with privacy-preserving federated learning for healthcare AI**
