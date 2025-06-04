"""
Model prediction utilities for OncoPredictAI

This module provides functions to make predictions with trained models.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded successfully")
    
    return model

def predict_kmeans(model, X):
    """
    Make predictions with a K-means clustering model.
    
    Args:
        model: Trained K-means model
        X: Feature matrix
        
    Returns:
        array: Predicted cluster labels
    """
    logger.info("Making predictions with K-means model")
    
    # Make predictions
    labels = model.predict(X)
    
    logger.info(f"Predictions made for {len(X)} samples")
    
    return labels

def predict_with_model(model, X, model_type=None):
    """
    Make predictions with any trained model.
    
    Args:
        model: Trained model
        X: Feature matrix
        model_type: Type of model (optional)
        
    Returns:
        array: Predictions
    """
    logger.info("Making predictions")
    
    # Make predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(X)
    else:
        logger.error("Model does not have a predict method")
        raise AttributeError("Model does not have a predict method")
    
    logger.info(f"Predictions made for {len(X)} samples")
    
    return predictions

def predict_probabilities(model, X):
    """
    Make probability predictions with a classification model.
    
    Args:
        model: Trained classification model
        X: Feature matrix
        
    Returns:
        array: Predicted probabilities
    """
    logger.info("Making probability predictions")
    
    # Check if model supports probability predictions
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        logger.info(f"Probability predictions made for {len(X)} samples")
        return probabilities
    else:
        logger.warning("Model does not support probability predictions")
        return None

def predict_and_evaluate(model, X, y, metrics=None):
    """
    Make predictions and evaluate against true values.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True target values
        metrics: List of metrics to calculate
        
    Returns:
        dict: Evaluation metrics
    """
    from sklearn import metrics as sk_metrics
    
    logger.info("Making predictions and evaluating model")
    
    # Default metrics
    if metrics is None:
        # Try to determine if classification or regression
        unique_values = np.unique(y)
        if len(unique_values) < 10 or np.issubdtype(y.dtype, np.integer):
            # Classification
            metrics = ['accuracy', 'f1', 'precision', 'recall']
        else:
            # Regression
            metrics = ['r2', 'mse', 'mae']
    
    # Make predictions
    predictions = predict_with_model(model, X)
    
    # Calculate metrics
    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = sk_metrics.accuracy_score(y, predictions)
        elif metric == 'f1':
            results['f1'] = sk_metrics.f1_score(y, predictions, average='weighted')
        elif metric == 'precision':
            results['precision'] = sk_metrics.precision_score(y, predictions, average='weighted')
        elif metric == 'recall':
            results['recall'] = sk_metrics.recall_score(y, predictions, average='weighted')
        elif metric == 'r2':
            results['r2'] = sk_metrics.r2_score(y, predictions)
        elif metric == 'mse':
            results['mse'] = sk_metrics.mean_squared_error(y, predictions)
        elif metric == 'mae':
            results['mae'] = sk_metrics.mean_absolute_error(y, predictions)
    
    # Log results
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return results
