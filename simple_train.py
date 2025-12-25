"""
Simple standalone training script without Flower
Trains model locally to generate a working model for the API
"""
import os
import sys
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

sys.path.insert(0, 'backend')

from backend.data_utils import download_liver_dataset, preprocess_data, save_test_data, get_data_loaders
from backend.model import create_model, train_model, evaluate_model, save_model

load_dotenv()

def simple_train():
    """Simple local training without federation"""
    print("=" * 60)
    print("SIMPLE LOCAL TRAINING MODE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\nLoading dataset...")
    X, y, feature_names = download_liver_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    save_test_data(X_test, y_test)
    
    # Step 2: Create data loaders
    print("\nPreparing data loaders...")
    train_loader = get_data_loaders(X_train, y_train, batch_size=32)
    test_loader = get_data_loaders(X_test, y_test, batch_size=32, shuffle=False)
    
    # Step 3: Create model
    print("\nCreating model...")
    model = create_model(input_dim=X_train.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Step 4: Train
    print(f"\nTraining{' on GPU' if torch.cuda.is_available() else ''}...")
    num_epochs = 20
    
    # Initialize training state
    os.makedirs('data', exist_ok=True)
    training_state = {
        'is_training': True,
        'current_round': 0,
        'total_rounds': num_epochs,
        'history': []
    }
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _, _ = evaluate_model(model, test_loader, criterion, device)
        
        # Update training state
        training_state['current_round'] = epoch + 1
        training_state['history'].append({
            'round': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'train_loss': float(train_loss),
                'train_accuracy': float(train_acc),
                'test_loss': float(test_loss),
                'test_accuracy': float(test_acc),
                'accuracy': float(test_acc),
                'loss': float(test_loss)
            }
        })
        
        # Write to file
        with open('data/training_state.json', 'w') as f:
            json.dump(training_state, f)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Mark training complete
    training_state['is_training'] = False
    with open('data/training_state.json', 'w') as f:
        json.dump(training_state, f)
    
    # Step 5: Save model
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    save_model(model, 'models/global_model_final.pth')
    
    print("\n" + "=" * 60)
    print(f"Training complete! Final Test Accuracy: {test_acc:.2%}")
    print("=" * 60)
    print(f"\nModel saved to: models/global_model_final.pth")
    print(f"Test Accuracy: {test_acc:.2%} ({int(test_acc * len(X_test))}/{len(X_test)} correct)")
    print("\nYou can now run predictions via the API!")
    
    return test_acc

if __name__ == '__main__':
    simple_train()
