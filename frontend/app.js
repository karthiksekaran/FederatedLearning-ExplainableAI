// API Configuration
const API_BASE_URL = window.location.origin;

// Global state
let featuresInfo = [];
let currentView = 'dashboard';
let pollingInterval = null;
let trainingStatusPolling = null;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeNavigation();
    loadFeatures();
    loadModelConfig();
    checkModelStatus();
    setupPredictionForm();
    startTrainingStatusPolling();
});

// Navigation
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);

            navButtons.forEach(b => {
                b.classList.remove('bg-indigo-600', 'text-white');
                b.classList.add('text-gray-700', 'hover:bg-gray-100');
            });
            btn.classList.add('bg-indigo-600', 'text-white');
            btn.classList.remove('text-gray-700', 'hover:bg-gray-100');
        });
    });
}

function switchView(viewName) {
    const views = document.querySelectorAll('.view');
    views.forEach(view => {
        view.classList.remove('active');
        view.classList.add('hidden');
    });
    const targetView = document.getElementById(viewName);
    targetView.classList.remove('hidden');
    targetView.classList.add('active');
    currentView = viewName;
}

// Load features information
async function loadFeatures() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/features`);
        const data = await response.json();

        if (data.features) {
            featuresInfo = data.features;
            generateFeatureInputs();
        }
    } catch (error) {
        console.error('Error loading features:', error);
        // Use default features if API fails
        featuresInfo = [
            { name: 'Age', min: 10, max: 90, mean: 45 },
            { name: 'Total_Bilirubin', min: 0.4, max: 75, mean: 3 },
            { name: 'Direct_Bilirubin', min: 0.1, max: 20, mean: 1.5 },
            { name: 'Alkaline_Phosphotase', min: 63, max: 2110, mean: 290 },
            { name: 'Alamine_Aminotransferase', min: 10, max: 2000, mean: 80 },
            { name: 'Aspartate_Aminotransferase', min: 10, max: 4929, mean: 110 },
            { name: 'Total_Proteins', min: 2.7, max: 9.6, mean: 6.5 },
            { name: 'Albumin', min: 0.9, max: 5.5, mean: 3.1 },
            { name: 'Albumin_Globulin_Ratio', min: 0.3, max: 2.8, mean: 0.9 },
            { name: 'Gender', min: 0, max: 1, mean: 0.5 }
        ];
        generateFeatureInputs();
    }
}

// Generate feature input fields
function generateFeatureInputs() {
    const container = document.getElementById('feature-inputs');
    container.innerHTML = '';

    featuresInfo.forEach(feature => {
        const formGroup = document.createElement('div');
        formGroup.className = 'space-y-2';

        const label = document.createElement('label');
        label.className = 'block text-sm font-medium text-gray-700';
        label.textContent = formatFeatureName(feature.name);
        label.htmlFor = `input-${feature.name}`;

        const input = document.createElement('input');
        input.className = 'w-full px-4 py-3 bg-white border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-colors';
        input.type = 'number';
        input.id = `input-${feature.name}`;
        input.name = feature.name;

        // Set appropriate step and decimal places
        if (feature.name === 'Age' || feature.name === 'Gender') {
            input.step = '1';
            input.value = Math.round(feature.mean);
        } else {
            input.step = '0.01';
            input.value = feature.mean.toFixed(2);
        }

        input.min = Math.floor(feature.min * 100) / 100;
        input.max = Math.ceil(feature.max * 100) / 100;
        input.placeholder = `Range: ${input.min} - ${input.max}`;

        formGroup.appendChild(label);
        formGroup.appendChild(input);
        container.appendChild(formGroup);
    });
}

// Format feature names for display
function formatFeatureName(name) {
    return name.replace(/_/g, ' ');
}

// Setup prediction form
function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await makePrediction();
    });
}

// Make prediction
async function makePrediction() {
    const formData = new FormData(document.getElementById('prediction-form'));
    const patientData = {};

    formData.forEach((value, key) => {
        patientData[key] = parseFloat(value);
    });

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ patient_data: patientData })
        });

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        displayPredictionResults(result);
    } catch (error) {
        console.error('Prediction error:', error);
        const errorMsg = error.message || 'Error making prediction';
        alert(`${errorMsg}. Please try again.`);
    }
}

// Display prediction results
function displayPredictionResults(result) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.style.display = 'block';

    // Update diagnosis
    const diagnosisText = document.getElementById('diagnosis-text');
    const confidenceText = document.getElementById('confidence-text');
    const resultIcon = document.getElementById('result-icon');

    diagnosisText.textContent = result.diagnosis;
    confidenceText.textContent = `Confidence: ${result.confidence.toFixed(1)}%`;

    if (result.prediction === 1) {
        resultIcon.textContent = '⚠️';
        resultIcon.style.color = '#EF4444';
    } else {
        resultIcon.textContent = '✅';
        resultIcon.style.color = '#10B981';
    }

    // Update probability bar
    const probabilityFill = document.getElementById('probability-fill');
    const probabilityValue = document.getElementById('probability-value');

    const displayProb = result.probability * 100;
    probabilityFill.style.width = `${displayProb}%`;
    probabilityValue.textContent = `${displayProb.toFixed(1)}%`;

    // Display SHAP values
    displaySHAPValues(result.explanation.top_features);

    // Display clinical interpretation
    const clinicalContent = document.getElementById('clinical-content');
    clinicalContent.innerHTML = formatClinicalText(result.clinical_interpretation);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display SHAP values
function displaySHAPValues(topFeatures) {
    const shapChart = document.getElementById('shap-chart');
    shapChart.innerHTML = '';

    topFeatures.forEach(([name, value]) => {
        const shapBar = document.createElement('div');
        shapBar.className = 'space-y-1';

        const label = document.createElement('div');
        label.className = 'flex items-center justify-between text-sm';
        label.innerHTML = `
            <span class="text-gray-700">${formatFeatureName(name)}</span>
            <span class="font-mono ${value >= 0 ? 'text-emerald-600' : 'text-red-600'}">${value >= 0 ? '+' : ''}${value.toFixed(3)}</span>
        `;

        const container = document.createElement('div');
        container.className = 'relative h-8 bg-gray-200 rounded-lg overflow-hidden';

        const fill = document.createElement('div');
        fill.className = 'absolute top-0 h-full transition-all duration-500';

        // Scale for visualization
        const maxAbsValue = Math.max(...topFeatures.map(([_, v]) => Math.abs(v)));
        const width = (Math.abs(value) / maxAbsValue) * 50;

        if (value >= 0) {
            fill.className += ' bg-gradient-to-r from-emerald-400 to-emerald-500';
            fill.style.left = '50%';
            fill.style.width = `${width}%`;
        } else {
            fill.className += ' bg-gradient-to-l from-red-400 to-red-500';
            fill.style.right = '50%';
            fill.style.width = `${width}%`;
        }

        // Center line
        const centerLine = document.createElement('div');
        centerLine.className = 'absolute left-1/2 top-0 w-px h-full bg-gray-400';

        container.appendChild(fill);
        container.appendChild(centerLine);
        shapBar.appendChild(label);
        shapBar.appendChild(container);
        shapChart.appendChild(shapBar);
    });
}

// Format clinical text with markdown-like formatting
function formatClinicalText(text) {
    // Bold text between **
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Line breaks
    text = text.replace(/\n\n/g, '</p><p>');
    text = text.replace(/\n/g, '<br>');
    return `<p>${text}</p>`;
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model/info`);
        const data = await response.json();

        if (data.exists) {
            updateTrainingStatus('Model Ready', 'completed');
        } else {
            updateTrainingStatus('Model Not Trained', 'pending');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
}

// Load model configuration
async function loadModelConfig() {
    try {
        const config = await fetchAPI('/api/config');
        displayModelConfig(config);
    } catch (error) {
        console.error('Error loading model config:', error);
    }
}

// Display model configuration
function displayModelConfig(config) {
    const modelInfoContainer = document.getElementById('model-info');

    if (!config || config.error) {
        modelInfoContainer.innerHTML = '<div class="bg-gray-50 p-4 rounded-lg border border-gray-200 col-span-full text-center"><p class="text-gray-600">Configuration unavailable</p></div>';
        return;
    }

    const model = config.model;
    const training = config.training;
    const dataset = config.dataset;

    modelInfoContainer.innerHTML = `
        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:border-indigo-300 transition-colors">
            <p class="text-xs text-gray-600 uppercase tracking-wider mb-2">Model</p>
            <p class="text-lg font-bold text-gray-900 font-mono">${model.name}</p>
            <p class="text-sm text-gray-600 mt-1">${model.framework} &bull; ${model.type}</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:border-emerald-300 transition-colors">
            <p class="text-xs text-gray-600 uppercase tracking-wider mb-2">Architecture</p>
            <p class="text-lg font-bold text-emerald-600 font-mono">${model.architecture.input_dim}&rarr;${model.architecture.hidden_layers.join('&rarr;')}&rarr;${model.architecture.output_dim}</p>
            <p class="text-sm text-gray-600 mt-1">${model.architecture.activation} + ${model.architecture.output_activation}</p>
        </div>
        ${model.total_parameters ? `
        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:border-purple-300 transition-colors">
            <p class="text-xs text-gray-600 uppercase tracking-wider mb-2">Parameters</p>
            <p class="text-lg font-bold text-purple-600 font-mono">${model.total_parameters.toLocaleString()}</p>
            <p class="text-sm text-gray-600 mt-1">${model.trainable_parameters.toLocaleString()} trainable</p>
        </div>
        ` : ''}
        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:border-amber-300 transition-colors">
            <p class="text-xs text-gray-600 uppercase tracking-wider mb-2">Training</p>
            <p class="text-lg font-bold text-amber-600 font-mono">${training.num_rounds} Rounds</p>
            <p class="text-sm text-gray-600 mt-1">${training.num_clients} clients &bull; ${training.epochs_per_round} epochs</p>
        </div>
        <div class="bg-gray-50 p-4 rounded-lg border border-gray-200 hover:border-teal-300 transition-colors">
            <p class="text-xs text-gray-600 uppercase tracking-wider mb-2">Dataset</p>
            <p class="text-lg font-bold text-teal-600 font-mono">ILPD</p>
            <p class="text-sm text-gray-600 mt-1">${dataset.features} features &bull; ${dataset.task}</p>
        </div>
    `;
}

// Start polling for training status
function startTrainingStatusPolling() {
    // Poll every 3 seconds
    trainingStatusPolling = setInterval(async () => {
        try {
            const status = await fetchAPI('/api/status');
            updateTrainingDisplay(status);
        } catch (error) {
            console.error('Error polling training status:', error);
        }
    }, 3000);

    // Also check immediately
    fetchAPI('/api/status').then(updateTrainingDisplay).catch(console.error);
}

// Update training display with status
function updateTrainingDisplay(status) {
    if (!status) return;

    const roundProgress = document.getElementById('round-progress');
    const progressFill = document.getElementById('progress-fill');

    if (status.is_training) {
        updateTrainingStatus('Training in Progress', 'training');
        roundProgress.textContent = `${status.current_round}/${status.total_rounds}`;
        const progress = (status.current_round / status.total_rounds) * 100;
        progressFill.style.width = `${progress}%`;
    } else if (status.current_round > 0) {
        updateTrainingStatus('Training Complete', 'completed');
        roundProgress.textContent = `${status.current_round}/${status.total_rounds}`;
        progressFill.style.width = '100%';
    }

    // Update client metrics if available
    if (status.history && status.history.length > 0) {
        displayTrainingHistory(status.history);
    }
}

// Display training history
function displayTrainingHistory(history) {
    const clientMetrics = document.getElementById('client-metrics');

    if (history.length === 0) return;

    const latestRound = history[history.length - 1];
    if (!latestRound.metrics) return;

    const metrics = latestRound.metrics;

    clientMetrics.innerHTML = `
        <div class="bg-gray-50 p-6 rounded-lg border border-gray-200">
            <h4 class="text-lg font-bold text-gray-900 mb-4">Round ${latestRound.round}</h4>
            <div class="space-y-3">
                <div class="flex items-center justify-between">
                    <span class="text-gray-600">Test Accuracy</span>
                    <span class="text-emerald-600 font-mono font-bold">${(metrics.accuracy * 100).toFixed(2)}%</span>
                </div>
                <div class="flex items-center justify-between">
                    <span class="text-gray-600">Test Loss</span>
                    <span class="text-amber-600 font-mono font-bold">${metrics.loss ? metrics.loss.toFixed(4) : 'N/A'}</span>
                </div>
                ${metrics.train_accuracy ? `
                <div class="flex items-center justify-between">
                    <span class="text-gray-600">Train Accuracy</span>
                    <span class="text-indigo-600 font-mono font-bold">${(metrics.train_accuracy * 100).toFixed(2)}%</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Update training status
function updateTrainingStatus(message, status) {
    const statusBadge = document.getElementById('training-status');

    let dotColor = 'bg-gray-400';
    if (status === 'completed') {
        dotColor = 'bg-emerald-500';
    } else if (status === 'training') {
        dotColor = 'bg-amber-500 animate-pulse';
    }

    statusBadge.innerHTML = `
        <div class="w-2 h-2 rounded-full ${dotColor}"></div>
        <span class="text-sm text-gray-700">${message}</span>
    `;
}

// Simulate training progress (for demonstration)
function simulateTraining(rounds = 5) {
    let currentRound = 0;
    const roundProgress = document.getElementById('round-progress');
    const progressFill = document.getElementById('progress-fill');

    const interval = setInterval(() => {
        currentRound++;
        roundProgress.textContent = `${currentRound}/${rounds}`;
        progressFill.style.width = `${(currentRound / rounds) * 100}%`;

        if (currentRound >= rounds) {
            clearInterval(interval);
            updateTrainingStatus('Training Complete', 'completed');
        }
    }, 2000);

    updateTrainingStatus('Training in Progress', 'training');
}

// Helper: Fetch with error handling
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}
