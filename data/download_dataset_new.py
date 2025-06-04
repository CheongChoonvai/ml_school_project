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
            
            # Count total images and analyze directories
            total_images = 0
            class_counts = {}
            
            for root, dirs, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 4 * level
                folder = os.path.basename(root)
                
                # Count image files (common image extensions)
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff'))]
                file_count = len(image_files)
                total_images += file_count
                
                if level <= 3:  # Limit the depth to avoid too much output
                    if file_count > 0:
                        print(f"{indent}{folder}/ [{file_count} images]")
                        # Track potential class information (useful for image classification datasets)
                        if level >= 2 and file_count > 10:  # Likely a class folder if it has many images
                            class_counts[folder] = file_count
                    else:
                        print(f"{indent}{folder}/")
            
            print(f"\nTotal images found: {total_images}")
            if class_counts:
                print("Potential classes detected:")
                for class_name, count in class_counts.items():
                    print(f"- {class_name}: {count} images")
    else:
        print(f"Path {path} is not a directory")

if __name__ == "__main__":
    try:
        print("OncoPredictAI Dataset Downloader")
        print("--------------------------------")
        print("Select dataset to download:")
        print("1. Global Cancer Patients dataset (2015-2024)")
        print("2. Chest X-ray Pneumonia dataset")
        print("3. Download both datasets")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1' or choice == '3':
            cancer_path = download_global_cancer_dataset()
            verify_dataset(cancer_path)
        
        if choice == '2' or choice == '3':
            xray_path = download_chest_xray_dataset()
            verify_dataset(xray_path)
            
        print("\nDataset download and verification completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTo use kagglehub, you need to:")
        print("1. Install the package: pip install kagglehub")
        print("2. Set up Kaggle API credentials:")
        print("   - Create an account on Kaggle")
        print("   - Go to Account settings and create an API token")
        print("   - Save the kaggle.json file to ~/.kaggle/kaggle.json")
        print("3. Make sure you have permissions to access the dataset")
