# Sensor-Driven Deep Learning Model Project

This project is a starting point for developing deep learning models that predict outputs (e.g., torque) from sensor data. It is designed to simulate sensor input and train a neural network model using PyTorch. The overall structure is inspired by the RoboAnkle project and is ideal for projects that plan to integrate real-world IoT sensor data in the future.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Generation and Preprocessing](#data-generation-and-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Files Description](#files-description)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

This repository contains code to simulate sensor data, define a neural network for regression, and train/evaluate the model. It's structured to allow for easy future integration with real sensor hardware and IoT devices.

## Project Structure

```
project/
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── data/                   # Folder for storing simulated or real sensor data
│   └── simulated_data.csv  # (Optional) CSV file containing generated sensor data
└── src/                    # Source code files
    ├── __init__.py         # Package initializer
    ├── neural_network.py   # Neural network model definitions
    ├── data_preprocessing.py # Data simulation/cleaning and splitting functions
    ├── train.py            # Script for training the model
    ├── evaluate.py         # Script for evaluating the trained model
    └── utils.py            # Utility functions (saving/loading models, logging, etc.)
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/krishbhalodiya/Prosth-Ankle-Prediction-Model.git
   cd Prosth-Ankle-Prediction-Model
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Generation and Preprocessing

The `data_preprocessing.py` script includes functions for generating synthetic sensor data and splitting it into training and test sets. This simulated data will be used until you can collect real sensor data.

Example usage:
```python
from src.data_preprocessing import load_or_generate_data, prepare_datasets

# Load existing data or generate new data
data = load_or_generate_data(data_path='data/simulated_data.csv')

# Prepare train and test datasets
train_loader, test_loader, scaler = prepare_datasets(data, batch_size=32)
```

### Model Training

To train the model, run:

```bash
python src/train.py
```

This script:
- Loads (or generates) the sensor data.
- Initializes the neural network (defined in `neural_network.py`).
- Trains the model on the generated data.
- Saves the trained model for later use.

### Model Evaluation

After training, you can evaluate the model by running:

```bash
python src/evaluate.py
```

This script:
- Loads the trained model.
- Runs inference on the test data.
- Plots the predicted outputs against the true outputs using matplotlib.

## Files Description

- **data_preprocessing.py**: Contains functions for generating synthetic sensor data, preprocessing the data, and creating PyTorch DataLoaders.
- **neural_network.py**: Defines the neural network architecture using PyTorch.
- **train.py**: Script for training the neural network model.
- **evaluate.py**: Script for evaluating the trained model on test data.
- **utils.py**: Utility functions for saving/loading models, logging training progress, etc.

## Future Work

- Integration with real sensor hardware
- Implementation of more advanced neural network architectures
- Development of a web-based dashboard for real-time monitoring
- Deployment to edge devices for real-time inference

