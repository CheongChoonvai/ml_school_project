"""
Test models functionality
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

class TestKMeans(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a small test dataset with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
        cluster2 = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
        cluster3 = np.random.randn(20, 2) * 0.5 + np.array([-5, -5])
        self.X = np.vstack([cluster1, cluster2, cluster3])
    
    def test_kmeans_clustering(self):
        """Test K-means clustering"""
        try:
            from models.clustering.kmeans import KMeans
            
            # Initialize and fit K-means
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(self.X)
            
            # Test that labels are assigned to all points
            self.assertEqual(len(kmeans.labels), len(self.X))
            
            # Test that there are exactly 3 clusters
            self.assertEqual(len(np.unique(kmeans.labels)), 3)
            
            # Test predict function
            test_point = np.array([[5, 5]])  # Should be closest to cluster 1
            label = kmeans.predict(test_point)[0]
            self.assertEqual(label, kmeans.predict(cluster1[0:1])[0])
            
        except ImportError:
            self.skipTest("KMeans model not available")

class TestPCA(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create a small test dataset with correlations
        np.random.seed(42)
        self.X = np.random.randn(50, 5)
        self.X[:, 0] = self.X[:, 1] + self.X[:, 2] + np.random.randn(50) * 0.1
    
    def test_pca_dimensionality_reduction(self):
        """Test PCA dimensionality reduction"""
        try:
            from models.dimensionality_reduction.pca import PCA
            
            # Initialize and fit PCA
            pca = PCA(n_components=2)
            X_transformed = pca.fit_transform(self.X)
            
            # Test that the transformed data has the right shape
            self.assertEqual(X_transformed.shape, (50, 2))
            
            # Test that the explained variance is computed
            self.assertIsNotNone(pca.explained_variance_ratio)
            self.assertEqual(len(pca.explained_variance_ratio), 2)
            
            # Test inverse transform
            X_reconstructed = pca.inverse_transform(X_transformed)
            self.assertEqual(X_reconstructed.shape, self.X.shape)
            
        except ImportError:
            self.skipTest("PCA model not available")

if __name__ == '__main__':
    unittest.main()
