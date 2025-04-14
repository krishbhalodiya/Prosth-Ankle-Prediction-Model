import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .neural_network import create_model
from .data_preprocessing import load_or_generate_data, prepare_datasets
from .utils import load_model

def evaluate_model(model, test_loader, device='cpu'):
    # Evaluate the model on test data and return predictions, targets, and metrics
    model.eval()
    model = model.to(device)
    
    # Lists to store predictions and targets
    all_predictions = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Convert lists to numpy arrays
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return predictions, targets, metrics

def plot_predictions(predictions, targets, save_path=None):
    # Plot predictions vs targets
    # Flatten arrays if they are not already 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predictions vs targets over samples
    ax1.plot(targets, label='True Values', alpha=0.7)
    ax1.plot(predictions, label='Predictions', alpha=0.7, linestyle='--')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of predictions vs targets
    ax2.scatter(targets, predictions, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(targets), np.min(predictions))
    max_val = max(np.max(targets), np.max(predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Predictions vs True Values Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    # Main function to run the evaluation process
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained neural network on sensor data')
    parser.add_argument('--data_path', type=str, default='data/simulated_data.csv',
                        help='Path to the CSV file containing sensor data')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                        help='Path to the trained model file')
    parser.add_argument('--hidden_dims', type=str, default='64,32',
                        help='Comma-separated list of hidden layer dimensions (must match trained model)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (must match trained model)')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for evaluation if available')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Path to save the evaluation plot')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load data
    data = load_or_generate_data(data_path=args.data_path)
    
    # Prepare datasets
    _, test_loader, _ = prepare_datasets(
        data=data,
        batch_size=args.batch_size,
        test_size=0.2  # Use same test size as during training
    )
    
    # Create model with same architecture as trained model
    input_dim = data.shape[1] - 1  # Number of features (excluding target)
    model = create_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout_rate=args.dropout
    )
    
    # Load trained model weights
    model = load_model(model, args.model_path)
    
    # Evaluate the model
    predictions, targets, metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.6f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Plot predictions vs targets
    plot_predictions(predictions, targets, args.save_plot)

if __name__ == "__main__":
    main() 