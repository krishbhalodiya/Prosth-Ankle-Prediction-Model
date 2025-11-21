# Prosthetic Ankle Stability Prediction Model

Real-time stability classification system using IMU sensor data from an M5Stack device. This project detects whether a user is walking stably or experiencing instability (stumbling, loss of balance) using machine learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Model Training](#3-model-training)
  - [4. Visualization](#4-visualization)
  - [5. Real-Time Inference](#5-real-time-inference)
- [Files Description](#files-description)
- [How It Works](#how-it-works)
- [Future Work](#future-work)

## Project Overview

This system uses Inertial Measurement Unit (IMU) data from an M5Stack device to classify stability in real-time. The goal is to detect instability events that could indicate a fall risk for prosthetic ankle users or individuals with mobility challenges.

**Key Features:**
- Real-time data capture from M5Stack via USB serial
- Binary classification: Stable vs Unstable
- Random Forest classifier with 100% accuracy on test data
- Live inference with color-coded console output
- Feature visualization tools

## Project Structure

```
Prosth-Ankle-Prediction-Model/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Raw sensor recordings
│   │   ├── stable_01.csv
│   │   └── unstable_01.csv
│   └── processed/              # Cleaned data and visualizations
│       ├── feature_clusters.png
│       └── raw_comparison.png
├── models/
│   └── stability_classifier.pkl  # Trained Random Forest model
└── src/
    ├── capture_sensor_data.py    # Capture data from M5Stack
    ├── clean_data.py             # Data preprocessing
    ├── train_classifier.py       # Train stability classifier
    ├── visualize_features.py     # Generate plots
    └── live_inference.py         # Real-time prediction
```

## Hardware Requirements

- **M5Stack** (Core, Core2, or similar) with IMU sensor
- USB cable
- Computer running macOS, Linux, or Windows

### M5Stack Code

The M5Stack should be running UIFlow code that prints IMU data in CSV format:

```python
# M5Stack UIFlow Code (simplified)
print("t_ms,ax,ay,az,gx,gy,gz,mx,my,mz")
while True:
    t_ms = time.ticks_ms()
    ax, ay, az = Imu.getAccel()
    gx, gy, gz = Imu.getGyro()
    mx, my, mz = Imu.getMag()
    print(f"{t_ms},{ax},{ay},{az},{gx},{gy},{gz},{mx},{my},{mz}")
    time.sleep(0.02)  # ~50 Hz
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/krishbhalodiya/Prosth-Ankle-Prediction-Model.git
   cd Prosth-Ankle-Prediction-Model
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection

Connect your M5Stack via USB and capture sensor data:

```bash
python src/capture_sensor_data.py
```

- Select the correct serial port (usually `/dev/cu.usbserial-*` on Mac)
- Walk normally for ~60 seconds → This is your **Stable** data
- Save the file as `data/raw/stable_01.csv`
- Repeat while simulating instability (stumbling, shaking) → **Unstable** data
- Save as `data/raw/unstable_01.csv`

**Note:** Make sure UIFlow is closed before running the capture script (only one program can access the serial port at a time).

### 2. Data Cleaning

Clean the raw data (removes empty magnetometer columns, resets timestamps):

```bash
python src/clean_data.py          # Cleans most recent file
python src/clean_data.py --all    # Cleans all files in data/raw/
```

### 3. Model Training

Train the Random Forest classifier:

```bash
python src/train_classifier.py
```

This will:
- Load `stable_01.csv` and `unstable_01.csv`
- Extract features from 50-sample windows (~1.5 seconds)
- Train a Random Forest with 100 trees
- Save the model to `models/stability_classifier.pkl`

Expected output: ~100% accuracy (stable walking is very distinct from wild shaking)

### 4. Visualization

Generate plots to understand the data:

```bash
python src/visualize_features.py
```

Creates:
- `data/processed/raw_comparison.png` - Raw acceleration over time
- `data/processed/feature_clusters.png` - Feature space scatter plot

### 5. Real-Time Inference

Run live predictions on streaming sensor data:

```bash
python src/live_inference.py
```

- Select your M5Stack port
- Walk around and see real-time predictions
- **Green** = STABLE, **Red** = UNSTABLE
- Press `Ctrl+C` to stop and see statistics

## Files Description

### Data Collection & Processing
- **`capture_sensor_data.py`**: Reads CSV data from M5Stack serial port, saves to timestamped files
- **`clean_data.py`**: Removes empty columns, resets time to start at 0

### Machine Learning
- **`train_classifier.py`**: Trains Random Forest on windowed features (mean, std, max, min, range)
- **`visualize_features.py`**: Plots raw data and feature space to show class separation

### Real-Time System
- **`live_inference.py`**: Loads trained model, buffers incoming sensor data, makes predictions every ~0.3s

## How It Works

### Algorithm Pipeline

1. **Windowing**: Incoming sensor data is grouped into 50-sample windows (~1.5 seconds at 30 Hz)
2. **Feature Extraction**: For each window, calculate statistics:
   - Mean, Standard Deviation, Max, Min, Range
   - Applied to all 6 sensor channels (ax, ay, az, gx, gy, gz)
3. **Classification**: Random Forest predicts `0` (Stable) or `1` (Unstable)
4. **Output**: Real-time console display with color-coded status

### Random Forest Classifier

- **100 Decision Trees** vote on each prediction
- Each tree learns a different pattern (e.g., "If gyro variance > 80, then Unstable")
- **Key Features**: Standard deviation and range of acceleration/gyroscope values
- **Why it works**: Unstable motion has high variance (erratic), stable walking has rhythmic patterns (low variance)

## Future Work

- [ ] Add second sensor on the foot for more accurate prosthetic simulation
- [ ] Implement feedback loop to M5Stack (change screen color on instability)
- [ ] Collect more diverse data (stairs, slopes, different walking speeds)
- [ ] Deploy model to M5Stack for edge inference (no laptop required)
- [ ] Web dashboard for remote monitoring
- [ ] Integration with physical prosthetic ankle hardware

## License

MIT License - Feel free to use this project for research or educational purposes.

---

**Author:** Krish Bhalodiya  
**Project Link:** [https://github.com/krishbhalodiya/Prosth-Ankle-Prediction-Model](https://github.com/krishbhalodiya/Prosth-Ankle-Prediction-Model)
