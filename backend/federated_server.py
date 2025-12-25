"""
Flower Federated Learning Server
Coordinates training and aggregates model updates
"""
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, Scalar
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import os
from dotenv import load_dotenv

from model import create_model, get_model_params, save_model, set_model_params, evaluate_model
from data_utils import get_data_loaders, load_test_data

load_dotenv()


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """Aggregate metrics using weighted average"""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(model, test_loader):
    """Return an evaluation function for server-side evaluation"""
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        """Evaluate global model on centralized test set"""
        # Set model parameters
        set_model_params(model, parameters)
        
        # Evaluate
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = torch.nn.BCELoss()
        
        loss, accuracy, _, _, _ = evaluate_model(model, test_loader, criterion, device)
        
        print(f"🌍 Round {server_round} - Global Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if server_round % 2 == 0:  # Save every 2 rounds
            save_model(model, f'models/global_model_round_{server_round}.pth')
        
        return loss, {"accuracy": accuracy}
    
    return evaluate


def start_server(num_rounds=5, num_clients=3, server_address='0.0.0.0:8080'):
    """Start Flower server with FedAvg strategy"""
    
    # Load test data for global evaluation
    X_test, y_test = load_test_data()
    if X_test is None:
        print("⚠️ No test data found. Run data preparation first.")
        return
    
    test_loader = get_data_loaders(X_test, y_test, batch_size=32, shuffle=False)
    
    # Create initial model
    model = create_model(input_dim=X_test.shape[1])
    initial_parameters = get_model_params(model)
    
    # Convert to Flower Parameters object
    initial_parameters_fl = fl.common.ndarrays_to_parameters(initial_parameters)
    
    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=num_clients,  # Minimum clients for training
        min_evaluate_clients=num_clients,  # Minimum clients for evaluation
        min_available_clients=num_clients,  # Minimum clients that need to connect
        evaluate_fn=get_evaluate_fn(model, test_loader),  # Server-side evaluation
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate client metrics
        initial_parameters=initial_parameters_fl,
    )
    
    print(f"🌍 Starting Flower server at {server_address}")
    print(f"🔄 Training for {num_rounds} rounds with {num_clients} clients")
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("✅ Federated training complete!")
    
    # Save final model
    save_model(model, 'models/global_model_final.pth')


if __name__ == '__main__':
    NUM_ROUNDS = int(os.getenv('NUM_ROUNDS', 5))
    NUM_CLIENTS = int(os.getenv('NUM_CLIENTS', 3))
    SERVER_PORT = os.getenv('FLOWER_SERVER_PORT', '8080')
    
    start_server(
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        server_address=f'0.0.0.0:{SERVER_PORT}'
    )
