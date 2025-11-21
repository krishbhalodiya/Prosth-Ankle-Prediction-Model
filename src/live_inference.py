import serial
import serial.tools.list_ports
import pandas as pd
import numpy as np
import joblib
import os
import sys
from collections import deque

# Configuration
BAUD_RATE = 115200
WINDOW_SIZE = 50  # Must match training
FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
MODEL_PATH = "models/stability_classifier.pkl"

class StabilityMonitor:
    def __init__(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_classifier.py first.")
        
        self.model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        
        # Buffer to store recent samples
        self.buffer = deque(maxlen=WINDOW_SIZE)
        
        # Statistics
        self.prediction_count = 0
        self.stable_count = 0
        self.unstable_count = 0
        
    def add_sample(self, sample_dict):
        """Add a new sample to the buffer"""
        self.buffer.append(sample_dict)
    
    def extract_features(self):
        """Extract features from the current buffer"""
        if len(self.buffer) < WINDOW_SIZE:
            return None  # Not enough data yet
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        
        # Calculate features
        features = {}
        for col in FEATURES:
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
        
        return pd.DataFrame([features])
    
    def predict(self):
        """Make a prediction on the current buffer"""
        features = self.extract_features()
        if features is None:
            return None
        
        prediction = self.model.predict(features)[0]
        self.prediction_count += 1
        
        if prediction == 0:
            self.stable_count += 1
            return "STABLE"
        else:
            self.unstable_count += 1
            return "UNSTABLE"
    
    def get_stats(self):
        """Return prediction statistics"""
        return {
            'total': self.prediction_count,
            'stable': self.stable_count,
            'unstable': self.unstable_count,
            'unstable_rate': self.unstable_count / self.prediction_count if self.prediction_count > 0 else 0
        }

def list_serial_ports():
    """Lists available serial ports."""
    ports = serial.tools.list_ports.comports()
    return ports

def select_port():
    """Lists ports and asks user to select one."""
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found. Please check your connection.")
        return None
    
    print("\nAvailable Serial Ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
    
    while True:
        try:
            selection = input("\nSelect port number (or 'q' to quit): ")
            if selection.lower() == 'q':
                return None
            index = int(selection) - 1
            if 0 <= index < len(ports):
                return ports[index].device
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def parse_line(line):
    """
    Parses a line of text from the sensor.
    Returns a dict of values if valid data, None otherwise.
    Expected format: t_ms,ax,ay,az,gx,gy,gz,mx,my,mz
    """
    try:
        line = line.strip()
        if not line:
            return None
        
        parts = line.split(',')
        
        if len(parts) != 10:
            return None
        
        # Parse values
        values = [float(x) for x in parts]
        
        # Return as dict (ignore magnetometer)
        return {
            't_ms': values[0],
            'ax': values[1],
            'ay': values[2],
            'az': values[3],
            'gx': values[4],
            'gy': values[5],
            'gz': values[6]
        }
    except ValueError:
        return None

def main():
    print("--- Real-Time Stability Monitor ---")
    
    # Load model
    try:
        monitor = StabilityMonitor(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Select port
    port = select_port()
    if not port:
        print("Exiting.")
        return

    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        
        # DTR/RTS toggle
        ser.dtr = False
        ser.rts = False
        import time
        time.sleep(0.1)
        ser.dtr = True
        ser.rts = True
        time.sleep(2)
        
        print("Connected!")
        print("\n--- Starting Real-Time Predictions ---")
        print("Press Ctrl+C to stop.\n")
        
        # Flush old data
        ser.reset_input_buffer()
        
        sample_count = 0
        
        while True:
            if ser.in_waiting > 0:
                try:
                    line_bytes = ser.readline()
                    line_str = line_bytes.decode('utf-8', errors='replace').strip()
                    
                    # Parse data
                    sample = parse_line(line_str)
                    
                    if sample:
                        monitor.add_sample(sample)
                        sample_count += 1
                        
                        # Make prediction every 10 samples (about 0.3s at 30Hz)
                        if sample_count % 10 == 0:
                            prediction = monitor.predict()
                            if prediction:
                                stats = monitor.get_stats()
                                
                                # Color output
                                if prediction == "STABLE":
                                    status = f"\033[92m{prediction}\033[0m"  # Green
                                else:
                                    status = f"\033[91m{prediction}\033[0m"  # Red
                                
                                sys.stdout.write(f"\r[Sample: {sample_count:5d}] Status: {status} | Unstable Rate: {stats['unstable_rate']*100:.1f}%   ")
                                sys.stdout.flush()
                                
                except UnicodeDecodeError:
                    pass
                    
    except serial.SerialException as e:
        print(f"\nError opening serial port: {e}")
    except KeyboardInterrupt:
        print(f"\n\n--- Session Summary ---")
        stats = monitor.get_stats()
        print(f"Total Predictions: {stats['total']}")
        print(f"Stable: {stats['stable']} ({stats['stable']/stats['total']*100:.1f}%)")
        print(f"Unstable: {stats['unstable']} ({stats['unstable_rate']*100:.1f}%)")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()

