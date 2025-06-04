"""
Test data processing functionality
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the project directory to the path
project_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(project_dir))

from src.data import preprocessing

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a small test DataFrame
        self.test_df = pd.DataFrame({
            'Age': [35, 42, 28, 50, np.nan],
            'Cancer_Type': ['Lung', 'Breast', 'Lung', 'Prostate', 'Breast'],
            'Genetic_Risk': [0.7, 0.4, 0.5, 0.8, 0.6],
            'Target_Severity_Score': [8, 5, 7, 9, 6]
        })
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Process the data
        df_processed = preprocessing.preprocess_data(self.test_df)
        
        # Check that missing values are handled
        self.assertEqual(df_processed['Age'].isna().sum(), 0)
        
        # Check that the DataFrame shape is preserved (no rows dropped)
        self.assertEqual(len(df_processed), len(self.test_df))
    
    def test_split_features_target(self):
        """Test splitting features and target"""
        # Split features and target
        X, y = preprocessing.split_features_target(self.test_df)
        
        # Check that target is not in features
        self.assertNotIn('Target_Severity_Score', X.columns)
        
        # Check that shapes are correct
        self.assertEqual(len(X), len(self.test_df))
        self.assertEqual(len(y), len(self.test_df))
    
    def test_handle_outliers(self):
        """Test outlier handling"""
        # Create a DataFrame with outliers
        df_outliers = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })
        
        # Process outliers using IQR method
        df_clean = preprocessing.handle_outliers(df_outliers, method='iqr')
        
        # Check that outlier is capped
        self.assertLess(df_clean['value'].max(), 100)

if __name__ == '__main__':
    unittest.main()
