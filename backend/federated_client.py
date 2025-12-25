"""
Flower Federated Learning Client
Implements NumPyClient for training on local hospital data
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import numpy as np

from model import create_model, get_model_params, set_model_params, train_model, evaluate_model
from data_utils import get_data_loaders


class LiverFederatedClient(fl.client.NumPyClient):
    """Federated Learning Client for Liver Disease Classification"""
    
    def __init__(self, client_id: int, X_train, y_train, X_val, y_val, 
                 batch_size=32, learning_rate=0.001, epochs=5):
        self.client_id = client_id
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Create data loaders
        self.train_loader = get_data_loaders(X_train, y_train, batch_size, shuffle=True)
        self.val_loader = get_data_loaders(X_val, y_val, batch_size, shuffle=False)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(input_dim=X_train.shape[1]).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"🏥 Client {client_id} initialized: {len(X_train)} train samples, {len(X_val)} val samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters"""
        return get_model_params(self.model)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model with provided parameters"""
        # Set model parameters from server
        set_model_params(self.model, parameters)
        
        # Train for specified epochs
        for epoch in range(self.epochs):
            train_loss, train_acc = train_model(
                self.model, self.train_loader, self.criterion, 
                self.optimizer, self.device
            )
        
        # Evaluate on validation set
        val_loss, val_acc, _, _, _ = evaluate_model(
            self.model, self.val_loader, self.criterion, self.device
        )
        
        print(f"🏥 Client {self.client_id} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Return updated parameters and metrics
        return get_model_params(self.model), len(self.train_loader.dataset), {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "client_id": self.client_id
        }
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model with provided parameters"""
        # Set model parameters from server
        set_model_params(self.model, parameters)
        
        # Evaluate
        val_loss, val_acc, _, _, _ = evaluate_model(
            self.model, self.val_loader, self.criterion, self.device
        )
        
        return val_loss, len(self.val_loader.dataset), {
            "accuracy": val_acc,
            "client_id": self.client_id
        }


def start_client(client_id: int, server_address: str, X_train, y_train, X_val, y_val,
                 batch_size=32, learning_rate=0.001, epochs=5):
    """Start a Flower client"""
    client = LiverFederatedClient(
        client_id, X_train, y_train, X_val, y_val,
        batch_size, learning_rate, epochs
    )
    
    # Start Flower client
    fl.client.start_client(
        server_address=server_address,
        client=client
    )


if __name__ == '__main__':
    # This will be called from run.py with appropriate arguments
    import sys
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Usage: python federated_client.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    server_address = os.getenv('FLOWER_SERVER_ADDRESS', '127.0.0.1:8080')
    
    # Load client data (this would be loaded from client-specific storage)
    # For now, we'll load it from the prepared splits
    import pickle
    with open(f'data/client_{client_id}_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    start_client(
        client_id=client_id,
        server_address=server_address,
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        batch_size=int(os.getenv('BATCH_SIZE', 32)),
        learning_rate=float(os.getenv('LEARNING_RATE', 0.001)),
        epochs=int(os.getenv('EPOCHS_PER_ROUND', 5))
    )
