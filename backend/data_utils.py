"""
Data utilities for downloading and preprocessing UCI Liver Disease dataset
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from ucimlrepo import fetch_ucirepo
import pickle


class LiverDataset(Dataset):
    """PyTorch Dataset for Liver Disease data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def download_liver_dataset(data_dir='data'):
    """Download UCI Liver Patient Dataset"""
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, 'liver_data.pkl')
    
    # Check if already downloaded
    if os.path.exists(cache_file):
        print("📦 Loading cached dataset...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y'], data['feature_names']
    
    print("📥 Downloading UCI Liver Patient Dataset...")
    try:
        # Fetch dataset (ID: 225 for ILPD)
        liver_patients = fetch_ucirepo(id=225)
        
        X = liver_patients.data.features
        y = liver_patients.data.targets
        
        # Convert gender to numeric FIRST (Male=1, Female=0)
        if 'Gender' in X.columns:
            X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
        
        # Handle missing values after gender conversion
        X = X.fillna(X.mean())
        
        # Convert to numpy
        X = X.values.astype(np.float32)
        y = y.values.ravel()
        
        # Binary classification: 1 (liver disease) vs 2 (no disease)
        # Convert to 1 (disease) and 0 (no disease)
        y = (y == 1).astype(np.float32)
        
        feature_names = liver_patients.data.features.columns.tolist()
        
        # Cache the data
        with open(cache_file, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'feature_names': feature_names}, f)
        
        print(f"✅ Dataset downloaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_names
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        # Fallback: Generate synthetic data for demo
        print("🔄 Generating synthetic data for demonstration...")
        return generate_synthetic_liver_data(data_dir)


def generate_synthetic_liver_data(data_dir='data'):
    """Generate synthetic liver disease data for demonstration"""
    np.random.seed(42)
    n_samples = 583  # Similar to actual dataset
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(np.float32)
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    cache_file = os.path.join(data_dir, 'liver_data.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'feature_names': feature_names}, f)
    
    print(f"✅ Synthetic dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, feature_names


def preprocess_data(X, y, train_size=0.8, random_seed=42):
    """Preprocess and split data"""
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler for later use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, X_test, y_train, y_test, scaler


def create_federated_splits(X_train, y_train, num_clients=3, random_seed=42):
    """Split training data into federated clients (simulating different hospitals)"""
    np.random.seed(random_seed)
    
    # Create non-IID splits (more realistic for healthcare)
    # Each hospital has different patient distributions
    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)
    
    # Create unequal splits (hospitals have different sizes)
    split_sizes = np.random.dirichlet(np.ones(num_clients) * 2, size=1)[0]
    split_indices = (np.cumsum(split_sizes) * n_samples).astype(int)
    
    client_data = []
    start_idx = 0
    
    for i in range(num_clients):
        end_idx = split_indices[i] if i < num_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]
        
        client_X = X_train[client_indices]
        client_y = y_train[client_indices]
        
        client_data.append({
            'X': client_X,
            'y': client_y,
            'size': len(client_X)
        })
        
        print(f"🏥 Client {i+1}: {len(client_X)} samples")
        start_idx = end_idx
    
    return client_data


def get_data_loaders(X, y, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader"""
    dataset = LiverDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def load_test_data():
    """Load preprocessed test data"""
    cache_file = 'data/test_data.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['X_test'], data['y_test']
    return None, None


def save_test_data(X_test, y_test):
    """Save test data for later evaluation"""
    os.makedirs('data', exist_ok=True)
    with open('data/test_data.pkl', 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)


if __name__ == '__main__':
    # Test data loading
    print("🧪 Testing data utilities...")
    X, y, feature_names = download_liver_dataset()
    print(f"\n📊 Dataset shape: {X.shape}")
    print(f"📊 Class distribution: {np.bincount(y.astype(int))}")
    print(f"📊 Features: {feature_names}")
    
    # Test preprocessing
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    print(f"\n✅ Train set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    
    # Test federated splits
    print("\n🔄 Creating federated splits...")
    client_data = create_federated_splits(X_train, y_train, num_clients=3)
    
    # Save test data
    save_test_data(X_test, y_test)
    print("\n✅ Data utilities test complete!")
