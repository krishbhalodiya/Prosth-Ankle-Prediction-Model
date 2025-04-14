import torch
import torch.nn as nn

class FeedforwardNet(nn.Module):
    # Simple feedforward network for regression tasks
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
        super(FeedforwardNet, self).__init__()
        
        layers = []
        
        # Input to first hidden layer
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def create_model(input_dim, hidden_dims=[64, 32], output_dim=1, dropout_rate=0.1):
    # Create and initialize a neural network model
    model = FeedforwardNet(input_dim, hidden_dims, output_dim, dropout_rate)
    
    # Initialize weights using Xavier/Glorot initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    return model 