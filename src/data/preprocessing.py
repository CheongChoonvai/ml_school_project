"""
Data preprocessing utilities for OncoPredictAI

This module provides functions to preprocess raw data for ML models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

def preprocess_data(df, config=None):
    """
    Preprocess raw cancer data.
    
    Args:
        df: Raw data DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame: Preprocessed data
    """
    logger.info("Starting data preprocessing")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Check for missing values
    missing_values = df_processed.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Found {missing_values.sum()} missing values")
        logger.info("Imputing missing values")
        
        # Separate numerical and categorical columns
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        # Impute numerical columns with median
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
        
        # Impute categorical columns with most frequent value
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
    
    # Handle categorical variables
    logger.info("Encoding categorical variables")
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # Use one-hot encoding for categories with few unique values
        if df_processed[col].nunique() < 10:
            one_hot = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, one_hot], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
        else:
            # For high cardinality, consider target encoding or other approaches
            logger.warning(f"Column {col} has high cardinality ({df_processed[col].nunique()} values)")
            # For now, we'll just label encode it
            df_processed[col] = df_processed[col].astype('category').cat.codes
    
    # Scale numerical features
    logger.info("Scaling numerical features")
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    logger.info("Data preprocessing completed")
    
    return df_processed

def split_features_target(df, target_col='Target_Severity_Score'):
    """
    Split data into features and target.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        
    Returns:
        tuple: (X, y) - features and target
    """
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame")
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def handle_outliers(df, method='iqr', threshold=1.5):
    """
    Detect and handle outliers in numerical columns.
    
    Args:
        df: DataFrame
        method: Method to detect outliers ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame: Data with outliers handled
    """
    df_clean = df.copy()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        if method == 'iqr':
            # IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Replace outliers with bounds
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            # Replace values beyond threshold standard deviations
            df_clean[col] = df_clean[col].clip(
                lower=mean - threshold * std,
                upper=mean + threshold * std
            )
    
    return df_clean
