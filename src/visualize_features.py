import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Reuse config from training
WINDOW_SIZE = 50
STEP_SIZE = 10
FEATURES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']

def load_data():
    stable_path = "data/raw/stable_01.csv"
    unstable_path = "data/raw/unstable_01.csv"
    
    if not os.path.exists(stable_path) or not os.path.exists(unstable_path):
        print("Data files not found.")
        return None, None
        
    df_stable = pd.read_csv(stable_path)
    df_unstable = pd.read_csv(unstable_path)
    
    return df_stable, df_unstable

def extract_features_df(df, label_name):
    """Helper to extract features for visualization"""
    features_list = []
    
    for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        window = df.iloc[i : i + WINDOW_SIZE]
        feats = {'label': label_name}
        for col in FEATURES:
            feats[f'{col}_std'] = window[col].std()
            feats[f'{col}_range'] = window[col].max() - window[col].min()
            feats[f'{col}_mean'] = window[col].mean()
        features_list.append(feats)
        
    return pd.DataFrame(features_list)

def plot_raw_comparison(df_stable, df_unstable):
    """Plots raw acceleration data comparison"""
    plt.figure(figsize=(14, 6))
    
    # Plot first 500 samples of 'ax'
    limit = 500
    plt.subplot(1, 2, 1)
    plt.plot(df_stable['ax'][:limit], label='Stable', color='blue')
    plt.title(f'Stable Walk (First {limit} samples)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Acceleration X (g)')
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_unstable['ax'][:limit], label='Unstable', color='red')
    plt.title(f'Unstable/Shake (First {limit} samples)')
    plt.xlabel('Time (samples)')
    plt.ylim(-2, 2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/raw_comparison.png')
    print("Saved raw_comparison.png")
    plt.close()

def plot_feature_clusters(feat_df):
    """Plots scatter plot of features to show separation"""
    plt.figure(figsize=(10, 8))
    
    # Scatter: Accel Std Dev vs Gyro Std Dev
    # These are usually the best indicators of stability
    sns.scatterplot(data=feat_df, x='ax_std', y='gx_std', hue='label', alpha=0.7, palette={'Stable': 'blue', 'Unstable': 'red'})
    
    plt.title('Feature Space: Acceleration Variance vs Gyro Variance')
    plt.xlabel('Acceleration X Std Dev (Shakiness)')
    plt.ylabel('Gyro X Std Dev (Rotation Speed)')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('data/processed/feature_clusters.png')
    print("Saved feature_clusters.png")
    plt.close()

def main():
    print("--- Visualizing Data & Features ---")
    df_stable, df_unstable = load_data()
    if df_stable is None: return

    # 1. Plot Raw Data
    plot_raw_comparison(df_stable, df_unstable)
    
    # 2. Extract Features for Scatter Plot
    print("Extracting features for plotting...")
    feats_stable = extract_features_df(df_stable, 'Stable')
    feats_unstable = extract_features_df(df_unstable, 'Unstable')
    
    all_feats = pd.concat([feats_stable, feats_unstable], ignore_index=True)
    
    # 3. Plot Clusters
    plot_feature_clusters(all_feats)
    print("\nDone! Check 'data/processed/' for the PNG images.")

if __name__ == "__main__":
    main()

