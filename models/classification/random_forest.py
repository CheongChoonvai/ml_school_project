"""
Random Forest Implementation for OncoPredictAI

This module provides a Random Forest implementation specifically
tailored for the OncoPredictAI Cancer Prediction system, with emphasis
on interpretability and resource efficiency for healthcare environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.utils import Bunch

logger = logging.getLogger(__name__)

class RandomForestModel:
    """
    Random Forest model for cancer data analysis, designed to be 
    resource-efficient and interpretable for healthcare staff.
    
    This class wraps scikit-learn's RandomForest implementation with
    additional features specifically for healthcare prediction tasks.
    """
    
    def __init__(self, task_type='classification', n_estimators=200, max_depth=6, 
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                 bootstrap=True, random_state=None, n_jobs=-1, class_weight=None):
        """
        Initialize the Random Forest model.
        
        Parameters:
        -----------
        task_type : str, default='classification'
            Type of task - 'classification' or 'regression'
            
        n_estimators : int, default=200
            Number of trees in the forest
            
        max_depth : int or None, default=6
            Maximum depth of the trees
            
        min_samples_split : int or float, default=2
            Minimum number of samples required to split a node
            
        min_samples_leaf : int or float, default=1
            Minimum number of samples required at a leaf node
            
        max_features : str or int or float, default='sqrt'
            Number of features to consider for the best split
            
        bootstrap : bool, default=True
            Whether to use bootstrap samples when building trees
            
        random_state : int or None, default=None
            Random seed for reproducibility
            
        n_jobs : int, default=-1
            Number of jobs to run in parallel
            
        class_weight : dict or 'balanced' or None, default=None
            Weights associated with classes for classification tasks
        """
        self.task_type = task_type
        
        # Store parameters
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        # Create the appropriate model based on task type
        if task_type == 'classification':
            self.params['class_weight'] = class_weight
            self.model = RandomForestClassifier(**self.params)
        elif task_type == 'regression':
            self.model = RandomForestRegressor(**self.params)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
            
        self.feature_names = None
        self.classes = None
        
        logger.info(f"Initialized {task_type} Random Forest model with {n_estimators} estimators")

    def fit(self, X, y, feature_names=None):
        """
        Fit the Random Forest model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        y : array-like of shape (n_samples,)
            Target values
            
        feature_names : list of str or None, default=None
            Names of the features
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        logger.info(f"Fitting Random Forest model to data with shape {X.shape}")
          # Save feature names if provided
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                raise ValueError("Length of feature_names doesn't match number of features in X")
            self.feature_names = feature_names
        
        # Fit the model
        self.model.fit(X, y)
        
        # Save classes for classification tasks
        if self.task_type == 'classification':
            # Only RandomForestClassifier has classes_ attribute
            self.classes = self.model.classes_
            logger.info(f"Model trained with {len(self.classes)} classes")
        else:
            # For regression, we don't have classes
            self.classes = None
          logger.info("Model training completed")
        return self

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
              logger.info(f"Making predictions on data with shape {X.shape}")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for classification task.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        y_pred_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba() is only available for classification tasks")
            
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
            
        # Make sure we're using a classifier that has predict_proba
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("The underlying model doesn't support predict_proba")
            
        logger.info(f"Predicting probabilities on data with shape {X.shape}")
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        y : array-like of shape (n_samples,)
            True values
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
            
        logger.info(f"Evaluating model on data with shape {X.shape}")
        
        y_pred = self.predict(X)
        
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
            
            # Only calculate ROC AUC for binary classification
            if len(np.unique(y)) == 2:
                y_pred_proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            
            # Create confusion matrix
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm
            
        elif self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, y_pred)
            
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def cross_validate(self, X, y, cv=5, scoring=None):
        """
        Perform cross-validation to evaluate the model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        y : array-like of shape (n_samples,)
            Target values
            
        cv : int, default=5
            Number of cross-validation folds
            
        scoring : str or None, default=None
            Scoring metric (uses 'accuracy' for classification, 'neg_mean_squared_error' for regression if None)
            
        Returns:
        --------
        cv_scores : array of shape (cv,)
            Cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Set default scoring if None
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        logger.info(f"Cross-validation {scoring} scores: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")
        return cv_scores

    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
        --------
        feature_importance : pandas.DataFrame
            DataFrame with feature names and importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
            
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create feature names if not provided during fit
        if self.feature_names is None:
            self.feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame with feature names and importances
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"Top 5 most important features: {feature_imp.head(5)['Feature'].tolist()}")
        return feature_imp

    def plot_feature_importance(self, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance scores.
        
        Parameters:
        -----------
        top_n : int or None, default=20
            Number of top features to display, or None for all features
            
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with feature importance plot
        """
        # Get feature importance DataFrame
        feature_imp = self.get_feature_importance()
        
        # Take top N features if specified
        if top_n is not None and top_n < len(feature_imp):
            feature_imp = feature_imp.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Random Forest Feature Importance')
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        return fig

    def plot_confusion_matrix(self, y_true, y_pred=None, figsize=(10, 8), class_names=None, normalize=False):
        """
        Plot confusion matrix for classification task.
        
        Parameters:
        -----------
        y_true : array-like of shape (n_samples,)
            True labels
            
        y_pred : array-like of shape (n_samples,) or None, default=None
            Predicted labels. If None, predictions will be made on the training data
            
        figsize : tuple, default=(10, 8)
            Figure size
            
        class_names : list or None, default=None
            List of class names
            
        normalize : bool, default=False
            Whether to normalize the confusion matrix
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure with confusion matrix plot
        """
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix is only available for classification tasks")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
            
        # Get predictions if not provided
        if y_pred is None:
            y_pred = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Get class names
        if class_names is None:
            class_names = [str(c) for c in self.classes] if self.classes is not None else [f"Class {i}" for i in range(len(cm))]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained yet. Call fit() first.")
            
        import joblib
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        model : RandomForestModel
            Loaded model
        """
        import joblib
        
        # Load model
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        
        return model


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load sample dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create and fit model
    rf = RandomForestModel(task_type='classification', n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train, feature_names=feature_names)
    
    # Evaluate model
    metrics = rf.evaluate(X_test, y_test)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 score: {metrics['f1_score']:.4f}")
    print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Plot feature importance
    rf.plot_feature_importance(top_n=10)
    plt.show()
    
    # Plot confusion matrix
    rf.plot_confusion_matrix(y_test, normalize=True, class_names=['Benign', 'Malignant'])
    plt.show()
