"""
Feature engineering for OncoPredictAI

This module provides functions for feature extraction and transformation.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def build_features(df, config=None):
    """
    Build features from preprocessed data.
    
    Args:
        df: Preprocessed DataFrame
        config: Configuration dictionary
        
    Returns:
        tuple: (features, target) - feature matrix and target vector
    """
    logger.info("Building features from preprocessed data")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Extract target if available
    target = None
    target_column = config.get('training', {}).get('target_column', 'Target_Severity_Score')
    if target_column in df_features.columns:
        logger.info(f"Extracting target column: {target_column}")
        target = df_features[target_column]
        df_features = df_features.drop(columns=[target_column])
    
    # Feature selection (if specified in config)
    selected_features = config.get('features', {}).get('selected_features', None)
    if selected_features:
        logger.info(f"Selecting {len(selected_features)} features")
        df_features = df_features[selected_features]
    
    # Feature engineering
    
    # 1. Add interaction terms for risk factors
    risk_factors = ['Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 'Smoking', 'Obesity_Level']
    risk_factors = [rf for rf in risk_factors if rf in df_features.columns]
    
    if len(risk_factors) >= 2:
        logger.info("Creating interaction features for risk factors")
        for i, factor1 in enumerate(risk_factors):
            for factor2 in risk_factors[i+1:]:
                interaction_name = f"{factor1}_{factor2}_interaction"
                df_features[interaction_name] = df_features[factor1] * df_features[factor2]
    
    # 2. Age-related features
    if 'Age' in df_features.columns:
        logger.info("Creating age-related features")
        df_features['Age_Squared'] = df_features['Age'] ** 2
        
        # Age groups
        bins = [0, 18, 35, 50, 65, 80, 100]
        labels = ['0-18', '19-35', '36-50', '51-65', '66-80', '81+']
        df_features['Age_Group'] = pd.cut(df_features['Age'], bins=bins, labels=labels, right=False)
        
        # One-hot encode age groups
        age_dummies = pd.get_dummies(df_features['Age_Group'], prefix='Age_Group')
        df_features = pd.concat([df_features, age_dummies], axis=1)
        df_features.drop('Age_Group', axis=1, inplace=True)
    
    # 3. Cancer type and stage features
    categorical_cols = ['Cancer_Type', 'Cancer_Stage']
    for col in categorical_cols:
        if col in df_features.columns and df_features[col].dtype == 'object':
            logger.info(f"One-hot encoding {col}")
            dummies = pd.get_dummies(df_features[col], prefix=col)
            df_features = pd.concat([df_features, dummies], axis=1)
            df_features.drop(col, axis=1, inplace=True)
    
    # Log transformation for skewed features
    skewed_features = config.get('features', {}).get('log_transform_features', [])
    for feature in skewed_features:
        if feature in df_features.columns:
            logger.info(f"Applying log transform to {feature}")
            # Add small constant to avoid log(0)
            df_features[f"{feature}_log"] = np.log1p(df_features[feature])
    
    logger.info(f"Feature building complete. Created {df_features.shape[1]} features")
    
    return df_features, target

def create_time_features(df, date_column):
    """
    Create time-based features from a date column.
    
    Args:
        df: DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with additional time features
    """
    if date_column not in df.columns:
        logger.warning(f"Date column {date_column} not found in DataFrame")
        return df
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Extract date components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    # Cyclic encoding for day of week and month
    df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[f'{date_column}_month'] / 12)
    df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[f'{date_column}_month'] / 12)
    df[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df[f'{date_column}_dayofweek'] / 7)
    df[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df[f'{date_column}_dayofweek'] / 7)
    
    return df
