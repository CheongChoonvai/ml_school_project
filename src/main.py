# Main entry point for running machine learning pipelines
"""
OncoPredictAI - Machine Learning Pipeline

This script serves as the main entry point for running the OncoPredictAI
machine learning pipelines. It provides functionality to:
1. Load and preprocess data
2. Train and evaluate different models
3. Generate visualizations
4. Save results

Usage:
    python main.py --config config/default.yaml --model kmeans

Author: OncoPredictAI Team
Date: June 2025
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add the project directory to the path
project_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(project_dir))

# Import project modules
from src.data import make_dataset, preprocessing
from src.features import build_features 
from src.models import train_model, predict_model
from src.visualization import visualize
from src.utils import evaluation, logging_config

def main(config_path, model_name=None, mode='train'):
    """
    Main function to run the machine learning pipeline.
    
    Args:
        config_path: Path to the configuration file
        model_name: Name of the model to run (kmeans, pca, random_forest, xgboost)
        mode: Mode to run (train, predict, evaluate)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure logging
    logging_config.setup_logging(config['logging'])
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline with model: {model_name}, mode: {mode}")
    
    # Data loading and preprocessing
    logger.info("Loading and preprocessing data")
    raw_data_path = config['data']['raw_cancer_data']
    df = make_dataset.load_data(raw_data_path)
    df_processed = preprocessing.preprocess_data(df, config)
    
    # Feature engineering
    logger.info("Building features")
    features, labels = build_features.build_features(df_processed, config)
    
    # Run the specified model
    if model_name:
        if model_name == 'kmeans':
            train_model.train_kmeans(features, config['models']['kmeans'])
        elif model_name == 'pca':
            train_model.train_pca(features, config['models']['pca'])
        elif model_name == 'random_forest':
            train_model.train_random_forest(features, labels, config['models']['random_forest'])
        elif model_name == 'xgboost':
            train_model.train_xgboost(features, labels, config['models']['xgboost'])
        else:
            logger.error(f"Unknown model: {model_name}")
            return
    else:
        logger.info("No model specified, running all models")
        # Run all models
        
    logger.info("Pipeline completed successfully")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run OncoPredictAI machine learning pipeline.')
    parser.add_argument('--config', type=str, default='config/default.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=False,
                        choices=['kmeans', 'pca', 'random_forest', 'xgboost'],
                        help='Name of the model to run')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'evaluate'],
                        help='Mode to run')
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.mode)
