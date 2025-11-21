import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SensorDataset(Dataset):
    # PyTorch Dataset for sensor data
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def generate_synthetic_data(n_samples=1000, n_features=6, noise_level=0.1, save_path=None):
    # Generate synthetic sensor data for simulation
    
    # Generate random sensor coefficients
    sensor_coefficients = np.random.uniform(-1, 1, size=n_features)
    
    # Generate random sensor data
    sensor_data = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Create synthetic output based on a function of the inputs
    targets = np.zeros(n_samples)
    for i in range(n_samples):
        # Linear combination of sensor readings
        linear_component = np.dot(sensor_data[i], sensor_coefficients)
        
        # Add nonlinearity
        nonlinear_component = 0.5 * np.sin(sensor_data[i, 0]) + 0.3 * (sensor_data[i, 1] * sensor_data[i, 2])
        
        targets[i] = linear_component + nonlinear_component
    
    # Add noise
    targets += np.random.normal(0, noise_level, n_samples)
    
    # Create DataFrame
    column_names = [f'sensor_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(sensor_data, columns=column_names)
    df['target'] = targets
    
    # Save to CSV if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return df

def prepare_datasets(data, test_size=0.2, batch_size=32, random_state=42):
    # Split data into training and test sets and create PyTorch DataLoaders
    
    # Extract features and target
    X = data.drop('target', axis=1).values
    y = data['target'].values.reshape(-1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets and data loaders
    train_dataset = SensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = SensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler

def load_or_generate_data(data_path='data/simulated_data.csv', n_samples=1000, n_features=6):
    # Load existing data or generate new data if file doesn't exist
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    else:
        print(f"Generating synthetic data...")
        return generate_synthetic_data(
            n_samples=n_samples, 
            n_features=n_features, 
            save_path=data_path
        ) 