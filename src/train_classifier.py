import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
WINDOW_SIZE = 50  # Number of samples per window (approx 1-2 seconds depending on Hz)
STEP_SIZE = 10    # Overlap: move window by 10 samples
FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_and_label_data(stable_path, unstable_path):
    """
    Loads raw CSVs, adds labels (0=Stable, 1=Unstable), and merges them.
    """
    print("Loading data...")
    
    # Load Stable
    if not os.path.exists(stable_path):
        print(f"Error: {stable_path} not found.")
        return None
    df_stable = pd.read_csv(stable_path)
    df_stable['label'] = 0
    print(f"  Stable samples: {len(df_stable)}")
    
    # Load Unstable
    if not os.path.exists(unstable_path):
        print(f"Error: {unstable_path} not found.")
        return None
    df_unstable = pd.read_csv(unstable_path)
    df_unstable['label'] = 1
    print(f"  Unstable samples: {len(df_unstable)}")
    
    # Remove empty columns if they exist
    drop_cols = ['mx', 'my', 'mz']
    df_stable = df_stable.drop(columns=[c for c in drop_cols if c in df_stable.columns])
    df_unstable = df_unstable.drop(columns=[c for c in drop_cols if c in df_unstable.columns])

    return pd.concat([df_stable, df_unstable], ignore_index=True)

def extract_features(df):
    """
    Converts raw time-series data into windowed features.
    Input: DataFrame with raw IMU data.
    Output: DataFrame with features (std, max, min, range) per window.
    """
    X = []
    y = []
    
    # We need to process stable and unstable sections separately so windows don't cross the boundary
    for label in [0, 1]:
        subset = df[df['label'] == label].reset_index(drop=True)
        
        # Sliding window
        for i in range(0, len(subset) - WINDOW_SIZE, STEP_SIZE):
            window = subset.iloc[i : i + WINDOW_SIZE]
            
            # Extract features for this window
            window_features = {}
            
            for col in FEATURES:
                # Statistical features are simple and effective for stability
                window_features[f'{col}_mean'] = window[col].mean()
                window_features[f'{col}_std']  = window[col].std()
                window_features[f'{col}_max']  = window[col].max()
                window_features[f'{col}_min']  = window[col].min()
                window_features[f'{col}_range'] = window_features[f'{col}_max'] - window_features[f'{col}_min']
            
            X.append(window_features)
            y.append(label)
            
    return pd.DataFrame(X), np.array(y)

def main():
    print("--- Training Stability Classifier ---")
    
    # Paths
    stable_file = "data/raw/stable_01.csv"
    unstable_file = "data/raw/unstable_01.csv"
    model_save_path = "models/stability_classifier.pkl"
    
    # 1. Load Data
    raw_df = load_and_label_data(stable_file, unstable_file)
    if raw_df is None:
        return

    # 2. Extract Features
    print(f"\nExtracting features (Window: {WINDOW_SIZE}, Step: {STEP_SIZE})...")
    X, y = extract_features(raw_df)
    print(f"  Created {len(X)} windows.")
    print(f"  Class balance: {np.bincount(y)} (0=Stable, 1=Unstable)")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train Model
    # Logistic Regression is fast and interpretable.
    # Random Forest is more powerful for complex nonlinear patterns.
    # We'll use Random Forest for better accuracy out of the box.
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Stable: {cm[0][0]} | False Unstable: {cm[0][1]}")
    print(f"False Stable: {cm[1][0]} | True Unstable: {cm[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stable', 'Unstable']))
    
    # Analyze misclassifications (if any)
    if acc < 1.0:
        print("\n--- Misclassified Windows Analysis ---")
        misclassified_idx = np.where(y_test != y_pred)[0]
        for idx in misclassified_idx:
            true_label = "Stable" if y_test[idx] == 0 else "Unstable"
            pred_label = "Stable" if y_pred[idx] == 0 else "Unstable"
            print(f"Window {idx}: True={true_label}, Pred={pred_label}")
            # Print a few key features to see why
            sample_features = X_test.iloc[idx]
            print(f"  ax_std: {sample_features['ax_std']:.4f}, gx_std: {sample_features['gx_std']:.4f}")
    else:
        print("\nNo misclassifications in the test set!")
        print("Note: Overlap in 2D plots doesn't mean overlap in high-dimensional space.")
    
    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()

