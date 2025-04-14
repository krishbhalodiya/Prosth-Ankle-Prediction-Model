import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
from tqdm import tqdm

from .neural_network import create_model
from .data_preprocessing import load_or_generate_data, prepare_datasets
from .utils import save_model, save_training_history, plot_training_history

def train(model, train_loader, val_loader=None, num_epochs=100, learning_rate=0.001, 
          device='cpu', model_dir='models', save_best=True):
    # Train the neural network model
    
    # Move model to device
    model = model.to(device)
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking training progress
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    # Start training
    print(f"Training started on {device}...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass, backward pass, and optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase if validation loader is provided
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # Save best model
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, os.path.join(model_dir, 'best_model.pt'))
                    print(f"Epoch {epoch+1}: Best model saved with validation loss: {val_loss:.6f}")
        
        # Print epoch results
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save the final model if not using save_best
    if not save_best or val_loader is None:
        save_model(model, os.path.join(model_dir, 'final_model.pt'))
    
    # Save training history
    save_training_history(history, os.path.join(model_dir, 'training_history.json'))
    
    return history

def main():
    # Main function to run the training process
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a neural network on sensor data')
    parser.add_argument('--data_path', type=str, default='data/simulated_data.csv',
                        help='Path to the CSV file containing sensor data')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate if data file does not exist')
    parser.add_argument('--n_features', type=int, default=6,
                        help='Number of features to generate if data file does not exist')
    parser.add_argument('--hidden_dims', type=str, default='64,32',
                        help='Comma-separated list of hidden layer dimensions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model and training history')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for training if available')
    parser.add_argument('--plot_history', action='store_true',
                        help='Plot training history after training')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load or generate data
    data = load_or_generate_data(
        data_path=args.data_path,
        n_samples=args.n_samples,
        n_features=args.n_features
    )
    
    # Prepare datasets
    train_loader, test_loader, _ = prepare_datasets(
        data=data,
        batch_size=args.batch_size
    )
    
    # Create model
    input_dim = data.shape[1] - 1  # Number of features (excluding target)
    model = create_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout_rate=args.dropout
    )
    
    # Train the model
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for simplicity
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        model_dir=args.model_dir
    )
    
    # Plot training history if requested
    if args.plot_history:
        plot_training_history(history, os.path.join(args.model_dir, 'training_history.png'))

if __name__ == "__main__":
    main() 