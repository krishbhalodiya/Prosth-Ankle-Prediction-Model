import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json

def save_model(model, filepath):
    # Save the trained model's state dictionary
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    # Load model state from a file
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"No model found at {filepath}")
        return model

def plot_training_history(history, save_path=None):
    # Plot training and validation loss history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def save_training_history(history, filepath):
    # Save training history to a JSON file
    # Convert numpy arrays or tensors to lists if present
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            history_serializable[key] = value.tolist()
        else:
            history_serializable[key] = value
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(history_serializable, f)
    
    print(f"Training history saved to {filepath}") 