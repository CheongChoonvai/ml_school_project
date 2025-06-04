"""
Dataset Download Script for OncoPredictAI

This script downloads the Global Cancer Patients dataset (2015-2024) and the Chest X-ray
Pneumonia dataset from Kaggle using the kagglehub library. This makes it easier to access
and update the datasets in a reproducible manner for cancer prediction and analysis.

Usage:
    python download_dataset.py

Requirements:
    - kagglehub package (install with: pip install kagglehub)
    - Kaggle API credentials configured
"""

import os
import kagglehub
import pandas as pd

def download_global_cancer_dataset():
    """
    Download the Global Cancer Patients dataset from Kaggle
    """
    print("Downloading Global Cancer Patients dataset (2015-2024)...")
    
    # Download latest version of the dataset
    path = kagglehub.dataset_download("zahidmughal2343/global-cancer-patients-2015-2024")
    
    print(f"Dataset downloaded to: {path}")
    return path

def download_chest_xray_dataset():
    """
    Download the Chest X-ray Pneumonia dataset from Kaggle
    """
    print("Downloading Chest X-ray Pneumonia dataset...")
    
    # Download latest version of the dataset
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print(f"Dataset downloaded to: {path}")
    return path

def verify_dataset(path):
    """
    Verify the downloaded dataset by loading it and displaying basic information
    """
    # If it's a directory, check for CSV files
    if os.path.isdir(path):
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in the dataset")
            
            # Load the primary dataset file
            main_file = os.path.join(path, csv_files[0])
            df = pd.read_csv(main_file)
            
            print("\nDataset Preview:")
            print(f"- Number of records: {len(df)}")
            print(f"- Number of features: {df.shape[1]}")
            print(f"- Features: {', '.join(df.columns.tolist())}")
            
            # Display basic statistics
            print("\nBasic statistics:")
            numeric_columns = df.select_dtypes(include=['number']).columns
            if len(numeric_columns) > 0:
                stats = df[numeric_columns].describe().T[['count', 'mean', 'min', 'max']]
                print(stats)
              # Save to standard location
            if "global-cancer-patients" in path:
                output_file = os.path.join(os.path.dirname(path), "global_cancer_patients_2015_2024.csv")
            else:
                # Use the name of the first CSV file if it's not the cancer dataset
                output_name = os.path.splitext(csv_files[0])[0] + ".csv"
                output_file = os.path.join(os.path.dirname(path), output_name)
                
            df.to_csv(output_file, index=False)
            print(f"\nDataset saved to: {output_file}")
        else:
            # For image datasets like chest X-ray
            print("\nThis appears to be an image dataset (no CSV files found)")
            print(f"Dataset structure:")
            for root, dirs, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * level
                folder = os.path.basename(root)
                if level <= 2:  # Limit the depth to avoid too much output
                    print(f"{indent}{folder}/")
                    if level == 2:
                        num_files = len(files)
                        if num_files > 0:
                            print(f"{indent}    [{num_files} files]")
    else:
        print(f"Path {path} is not a directory")

if __name__ == "__main__":
    try:
        print("Select dataset to download:")
        print("1. Global Cancer Patients dataset (2015-2024)")
        print("2. Chest X-ray Pneumonia dataset")
        print("3. Both datasets")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            path = download_global_cancer_dataset()
            verify_dataset(path)
        elif choice == '2':
            path = download_chest_xray_dataset()
            verify_dataset(path)
        elif choice == '3':
            path1 = download_global_cancer_dataset()
            verify_dataset(path1)
            
            print("\n" + "-" * 50 + "\n")
            
            path2 = download_chest_xray_dataset()
            verify_dataset(path2)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have kagglehub installed (pip install kagglehub)")
        print("2. Ensure you've set up Kaggle API credentials")
        print("   - Create a Kaggle account at https://www.kaggle.com")
        print("   - Go to 'Account' > 'Create API Token' to download kaggle.json")
        print("   - Place kaggle.json in ~/.kaggle/ directory (or %USERPROFILE%\\.kaggle\\ on Windows)")
        print("   - Ensure permissions are set correctly: chmod 600 ~/.kaggle/kaggle.json")
        print("3. Check your internet connection")
