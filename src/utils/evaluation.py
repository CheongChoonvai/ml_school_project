"""
Model evaluation utilities for OncoPredictAI

This module provides functions for model evaluation.
"""

import numpy as np
import pandas as pd
import logging
from sklearn import metrics

logger = logging.getLogger(__name__)

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating classification model")
    
    results = {}
    
    # Basic classification metrics
    results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    
    # Check if binary or multiclass
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(unique_labels) <= 2:
        # Binary classification
        results['precision'] = metrics.precision_score(y_true, y_pred)
        results['recall'] = metrics.recall_score(y_true, y_pred)
        results['f1'] = metrics.f1_score(y_true, y_pred)
        
        # AUC-ROC if probabilities are provided
        if y_prob is not None:
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                prob_positive = y_prob[:, 1]  # Probability of the positive class
            else:
                prob_positive = y_prob
            
            try:
                results['auc_roc'] = metrics.roc_auc_score(y_true, prob_positive)
                
                # ROC curve points
                fpr, tpr, _ = metrics.roc_curve(y_true, prob_positive)
                results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                
                # Precision-Recall curve points
                precision, recall, _ = metrics.precision_recall_curve(y_true, prob_positive)
                results['pr_curve'] = {'precision': precision, 'recall': recall}
                results['average_precision'] = metrics.average_precision_score(y_true, prob_positive)
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR metrics: {str(e)}")
    else:
        # Multiclass classification
        results['precision_macro'] = metrics.precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = metrics.recall_score(y_true, y_pred, average='macro')
        results['f1_macro'] = metrics.f1_score(y_true, y_pred, average='macro')
        
        results['precision_weighted'] = metrics.precision_score(y_true, y_pred, average='weighted')
        results['recall_weighted'] = metrics.recall_score(y_true, y_pred, average='weighted')
        results['f1_weighted'] = metrics.f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    results['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    
    # Classification report
    results['classification_report'] = metrics.classification_report(y_true, y_pred)
    
    # Log results
    logger.info(f"Classification metrics:")
    for key, value in {k: v for k, v in results.items() if not isinstance(v, dict) and not isinstance(v, np.ndarray)}.items():
        logger.info(f"  {key}: {value}")
    
    return results

def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating regression model")
    
    results = {}
    
    # Basic regression metrics
    results['r2'] = metrics.r2_score(y_true, y_pred)
    results['mae'] = metrics.mean_absolute_error(y_true, y_pred)
    results['mse'] = metrics.mean_squared_error(y_true, y_pred)
    results['rmse'] = np.sqrt(results['mse'])
    
    # Median absolute error
    results['median_ae'] = metrics.median_absolute_error(y_true, y_pred)
    
    # Max error
    results['max_error'] = metrics.max_error(y_true, y_pred)
    
    # Explained variance
    results['explained_variance'] = metrics.explained_variance_score(y_true, y_pred)
    
    # Log results
    logger.info(f"Regression metrics:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")
    
    return results

def evaluate_clustering_model(X, labels, model=None):
    """
    Evaluate clustering model performance.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        model: Clustering model (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    logger.info("Evaluating clustering model")
    
    results = {}
    
    try:
        # Calculate metrics
        results['silhouette'] = silhouette_score(X, labels)
        results['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        results['davies_bouldin'] = davies_bouldin_score(X, labels)
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        results['cluster_sizes'] = dict(zip([f'cluster_{i}' for i in unique_labels], counts))
        
        # Inertia (if model has it)
        if model is not None and hasattr(model, 'inertia_'):
            results['inertia'] = model.inertia_
        
        # Log results
        logger.info(f"Clustering metrics:")
        for key, value in {k: v for k, v in results.items() if not isinstance(v, dict)}.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Cluster sizes:")
        for cluster, size in results['cluster_sizes'].items():
            logger.info(f"  {cluster}: {size}")
    
    except Exception as e:
        logger.error(f"Error calculating clustering metrics: {str(e)}")
    
    return results

def cross_validate_model(model, X, y, cv=5, scoring=None):
    """
    Perform cross-validation of model.
    
    Args:
        model: Model to cross-validate
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: Scoring metrics
        
    Returns:
        dict: Cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    logger.info(f"Performing {cv}-fold cross-validation")
    
    # Default scoring based on problem type
    if scoring is None:
        # Try to determine if classification or regression
        unique_values = np.unique(y)
        if len(unique_values) < 10 or np.issubdtype(y.dtype, np.integer):
            # Classification
            if len(unique_values) <= 2:
                # Binary classification
                scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:
                # Multi-class classification
                scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        else:
            # Regression
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    # Process results
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores)
        }
    
    # Log results
    logger.info(f"Cross-validation results:")
    for metric, scores in results.items():
        logger.info(f"  {metric}:")
        logger.info(f"    Test: {scores['test_mean']:.4f} ± {scores['test_std']:.4f}")
        logger.info(f"    Train: {scores['train_mean']:.4f} ± {scores['train_std']:.4f}")
    
    return results
