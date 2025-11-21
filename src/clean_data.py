import pandas as pd
import os
import argparse
import glob

def clean_sensor_data(input_path, output_path=None):
    """
    Cleans raw M5Stack sensor data.
    1. Removes magnetometer columns (mx, my, mz) as they are empty.
    2. Resets timestamp so the first sample is at t=0.
    3. Converts t_ms to seconds for easier interpretation.
    """
    print(f"Processing: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 1. Remove Magnetometer columns if they exist
    drop_cols = ['mx', 'my', 'mz']
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"  - Dropped empty columns: {existing_drop}")

    # 2. Reset Time
    if 't_ms' in df.columns:
        start_time = df['t_ms'].iloc[0]
        df['time_sec'] = (df['t_ms'] - start_time) / 1000.0
        
        # Reorder columns to put time_sec first
        cols = ['time_sec'] + [c for c in df.columns if c not in ['t_ms', 'time_sec']]
        df = df[cols]
        print("  - Reset timestamps to start at 0.0s")
    
    # 3. Basic Validation
    print(f"  - Shape: {df.shape}")
    print(f"  - Duration: {df['time_sec'].max():.2f} seconds")

    # Determine output path
    if output_path is None:
        # Default: save to data/processed with same filename
        filename = os.path.basename(input_path)
        output_path = os.path.join("data", "processed", f"cleaned_{filename}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Clean raw sensor data.")
    parser.add_argument("file", nargs="?", help="Specific file to clean. If empty, cleans the most recent file in data/raw/")
    parser.add_argument("--all", action="store_true", help="Clean ALL files in data/raw/")
    
    args = parser.parse_args()
    
    raw_dir = os.path.join("data", "raw")
    
    if args.all:
        # Process all csv files in data/raw
        files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if not files:
            print("No files found in data/raw/")
            return
        for f in files:
            clean_sensor_data(f)
            
    elif args.file:
        # Process specific file
        clean_sensor_data(args.file)
        
    else:
        # Process most recent file by default
        files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
        if not files:
            print("No files found in data/raw/")
            return
        most_recent = files[-1]
        print("No file specified. Picking most recent:")
        clean_sensor_data(most_recent)

if __name__ == "__main__":
    main()

