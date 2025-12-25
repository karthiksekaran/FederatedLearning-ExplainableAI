"""
Neural Network model for liver disease classification
Optimized for federated learning (small, efficient)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiverDiseaseClassifier(nn.Module):
    """
    Binary classification model for liver disease prediction
    Lightweight architecture suitable for federated learning
    """
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32], dropout_rate=0.3):
        super(LiverDiseaseClassifier, self).__init__()
        
        self.input_dim = input_dim
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        """Return probability for positive class"""
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x, threshold=0.5):
        """Return binary predictions"""
        proba = self.predict_proba(x)
        return (proba > threshold).float()


def create_model(input_dim=10):
    """Factory function to create model"""
    return LiverDiseaseClassifier(input_dim=input_dim)


def get_model_params(model):
    """Extract model parameters as list of numpy arrays"""
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_params(model, params):
    """Set model parameters from list of numpy arrays"""
    state_dict = model.state_dict()
    for i, (key, param) in enumerate(zip(state_dict.keys(), params)):
        state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict, strict=True)


def train_model(model, train_loader, criterion, optimizer, device='cpu'):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_y = batch_y.unsqueeze(1)  # Add dimension for BCELoss
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device='cpu'):
    """Evaluate model on test data"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.unsqueeze(1)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probas.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels, all_probas


def save_model(model, path='models/global_model.pth'):
    """Save model to disk"""
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")


def load_model(path='models/global_model.pth', input_dim=10):
    """Load model from disk"""
    model = create_model(input_dim=input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"📂 Model loaded from {path}")
    return model


if __name__ == '__main__':
    # Test model
    print("🧪 Testing model...")
    
    # Create model
    model = create_model(input_dim=10)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(32, 10)
    output = model(dummy_input)
    print(f"✅ Forward pass: input {dummy_input.shape} -> output {output.shape}")
    
    # Test parameter extraction
    params = get_model_params(model)
    print(f"✅ Extracted {len(params)} parameter tensors")
    
    # Test parameter setting
    set_model_params(model, params)
    print(f"✅ Parameters set successfully")
    
    print("\n✅ Model test complete!")
