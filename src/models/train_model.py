"""
Model training utilities for OncoPredictAI

This module provides functions to train different machine learning models.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def train_kmeans(X, config):
    """
    Train a K-means clustering model.
    
    Args:
        X: Feature matrix
        config: Model configuration dictionary
        
    Returns:
        Trained K-means model
    """
    from models.clustering.kmeans import KMeans
    
    logger.info("Training K-means clustering model")
    
    # Extract hyperparameters from config
    n_clusters = config.get('n_clusters', 3)
    max_iters = config.get('max_iter', 300)
    tol = config.get('tol', 1e-4)
    random_state = config.get('random_state', 42)
    
    # Initialize and train model
    model = KMeans(
        n_clusters=n_clusters,
        max_iters=max_iters,
        tol=tol,
        random_state=random_state
    )
    
    # Fit the model
    model.fit(X)
    
    # Get cluster labels
    labels = model.labels
    
    # Log results
    logger.info(f"K-means clustering completed with {n_clusters} clusters")
    logger.info(f"Inertia: {model.inertia}")
    
    # Save model
    save_model(model, 'kmeans', 'clustering')
    
    return model

def train_pca(X, config):
    """
    Train a PCA model.
    
    Args:
        X: Feature matrix
        config: Model configuration dictionary
        
    Returns:
        Trained PCA model
    """
    from models.dimensionality_reduction.pca import PCA
    
    logger.info("Training PCA model")
    
    # Extract hyperparameters from config
    n_components = config.get('n_components', 2)
    
    # Initialize and train model
    model = PCA(n_components=n_components)
    
    # Fit the model and transform the data
    X_transformed = model.fit_transform(X)
    
    # Log results
    if model.explained_variance_ratio is not None:
        explained_variance = model.explained_variance_ratio
        logger.info(f"PCA completed with {n_components} components")
        logger.info(f"Explained variance ratios: {explained_variance}")
        logger.info(f"Cumulative explained variance: {np.sum(explained_variance)}")
    
    # Save model
    save_model(model, 'pca', 'dimensionality_reduction')
    
    return model, X_transformed

def train_random_forest(X, y, config):
    """
    Train a Random Forest model.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Model configuration dictionary
        
    Returns:
        Trained Random Forest model
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    
    logger.info("Training Random Forest model")
    
    # Extract hyperparameters from config
    n_estimators = config.get('n_estimators', 100)
    max_depth = config.get('max_depth', None)
    random_state = config.get('random_state', 42)
    
    # Split data into train and test sets
    test_size = config.get('test_size', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Determine if classification or regression task
    unique_values = np.unique(y)
    if len(unique_values) < 10 or np.issubdtype(y.dtype, np.integer):
        # Classification task
        logger.info("Training Random Forest Classifier")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Random Forest Classifier accuracy: {accuracy:.4f}")
        
    else:
        # Regression task
        logger.info("Training Random Forest Regressor")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"Random Forest Regressor R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Save model
    save_model(model, 'random_forest', 'classification')
    
    return model

def train_xgboost(X, y, config):
    """
    Train an XGBoost model.
    
    Args:
        X: Feature matrix
        y: Target vector
        config: Model configuration dictionary
        
    Returns:
        Trained XGBoost model
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
        
        logger.info("Training XGBoost model")
        
        # Extract hyperparameters from config
        max_depth = config.get('max_depth', 6)
        learning_rate = config.get('learning_rate', 0.05)
        n_estimators = config.get('n_estimators', 200)
        subsample = config.get('subsample', 0.8)
        colsample_bytree = config.get('colsample_bytree', 0.75)
        min_child_weight = config.get('min_child_weight', 3)
        objective = config.get('objective', 'binary:logistic')
        
        # Split data into train and test sets
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Determine if classification or regression task
        unique_values = np.unique(y)
        if len(unique_values) < 10 or np.issubdtype(y.dtype, np.integer):
            # Classification task
            logger.info("Training XGBoost Classifier")
            if len(unique_values) == 2:
                # Binary classification
                objective = 'binary:logistic'
            else:
                # Multi-class classification
                objective = 'multi:softprob'
                
            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                objective=objective,
                random_state=random_state
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"XGBoost Classifier accuracy: {accuracy:.4f}")
            
        else:
            # Regression task
            logger.info("Training XGBoost Regressor")
            model = xgb.XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                objective='reg:squarederror',
                random_state=random_state
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"XGBoost Regressor R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # Save model
        save_model(model, 'xgboost', 'classification')
        
        return model
        
    except ImportError:
        logger.error("XGBoost not installed. Please install it with: pip install xgboost")
        raise

def save_model(model, model_name, model_type):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        model_type: Type of model (clustering, classification, etc.)
        
    Returns:
        str: Path to saved model
    """
    # Create directory if it doesn't exist
    model_dir = os.path.join('outputs', 'models', 'serialized')
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")
    
    return filepath
