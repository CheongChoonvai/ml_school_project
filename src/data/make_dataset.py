"""
Data loading utilities for OncoPredictAI

This module provides functions to load raw data from different sources.
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load data from a file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame: Loaded data
    """
    logger.info(f"Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_extension = os.path.splitext(filepath)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif file_extension == '.json':
        df = pd.read_json(filepath)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    return df

def download_cancer_data():
    """
    Download cancer data from Kaggle.
    
    Returns:
        str: Path to downloaded data
    """
    try:
        import kagglehub
        
        logger.info("Downloading Global Cancer Patients dataset from Kaggle")
        
        path = kagglehub.dataset_download(
            "zahidmughal2343/global-cancer-patients-2015-2024"
        )
        
        logger.info(f"Downloaded dataset to: {path}")
        
        return path
    
    except ImportError:
        logger.error("kagglehub not installed. Please install it with: pip install kagglehub")
        raise
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def download_xray_data():
    """
    Download X-ray data from Kaggle.
    
    Returns:
        str: Path to downloaded data
    """
    try:
        import kagglehub
        
        logger.info("Downloading Chest X-ray Pneumonia dataset from Kaggle")
        
        path = kagglehub.dataset_download(
            "paultimothymooney/chest-xray-pneumonia"
        )
        
        logger.info(f"Downloaded dataset to: {path}")
        
        return path
    
    except ImportError:
        logger.error("kagglehub not installed. Please install it with: pip install kagglehub")
        raise
    
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    download_cancer_data()
    download_xray_data()
