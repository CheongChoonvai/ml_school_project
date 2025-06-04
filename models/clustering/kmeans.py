"""
K-means Clustering Implementation from Scratch
Designed for the OncoPredictAI Cancer Analysis System

This implementation is optimized for computational efficiency
with limited hardware resources, focusing on interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class KMeans:
    """
    Custom K-means implementation designed for cancer data analysis.
    
    This implementation can work effectively with limited computing resources,
    handles noisy data common in medical datasets, and provides
    visualizations that are interpretable to healthcare staff.
    """
    
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=None):
        """
        Initialize KMeans with the number of clusters and other parameters.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form.
            
        max_iters : int, default=300
            Maximum number of iterations for a single run.
            
        tol : float, default=1e-4
            Tolerance for declaring convergence.
            
        random_state : int or None, default=None
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        
    def _initialize_centroids(self, X):
        """
        Initialize centroids using K-means++ algorithm for better convergence.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        centroids : numpy array, shape (n_clusters, n_features)
            Initial centroids.
        """
        np.random.seed(self.random_state)
        
        # Choose first centroid randomly
        centroids = [X[np.random.choice(X.shape[0])]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances from points to the centroids
            dist = np.min(cdist(X, np.array(centroids), 'euclidean'), axis=1)
            
            # Select remaining points based on their distances (probability proportional to distance squared)
            probs = dist**2 / np.sum(dist**2)
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            ind = np.where(cumprobs >= r)[0][0]
            centroids.append(X[ind])
            
        return np.array(centroids)
    
    def fit(self, X):
        """
        Fit the KMeans model to the data.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data to fit the KMeans model.
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        prev_centroids = np.zeros_like(self.centroids)
        
        # Iterate until convergence or max iterations
        for _ in range(self.max_iters):
            # Calculate distances between data points and centroids
            distances = cdist(X, self.centroids, 'euclidean')
            
            # Assign each point to the nearest centroid
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids based on mean of assigned points
            for i in range(self.n_clusters):
                if np.sum(self.labels == i) > 0:  # Check if the cluster has points
                    self.centroids[i] = np.mean(X[self.labels == i], axis=0)
            
            # Check for convergence
            if np.sum((self.centroids - prev_centroids) ** 2) < self.tol:
                break
                
            # Update previous centroids
            prev_centroids = self.centroids.copy()
        
        # Calculate inertia (sum of squared distances to centroids)
        self.inertia = self._calculate_inertia(X)
        
        return self
    
    def _calculate_inertia(self, X):
        """
        Calculate the inertia (within-cluster sum of squares).
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        inertia : float
            Sum of squared distances of samples to their closest centroid.
        """
        distances = cdist(X, self.centroids, 'euclidean')
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)
    
    def fit_predict(self, X):
        """
        Fit the KMeans model and return the cluster labels.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data to fit the KMeans model.
            
        Returns:
        --------
        labels : numpy array, shape (n_samples,)
            Cluster labels for each point in the training data.
        """
        self.fit(X)
        return self.labels
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            New data to predict.
            
        Returns:
        --------
        labels : numpy array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.centroids is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Calculate distances between data points and centroids
        distances = cdist(X, self.centroids, 'euclidean')
        
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)
    
    def calculate_silhouette_score(self, X):
        """
        Calculate the mean Silhouette Coefficient for all samples.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The data to evaluate.
            
        Returns:
        --------
        silhouette_score : float
            Mean Silhouette Coefficient for all samples.
        """
        if self.centroids is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Get cluster labels
        if self.labels is None:
            self.labels = self.predict(X)
            
        # Calculate pairwise distances
        pairwise_distances = cdist(X, X, 'euclidean')
        
        # Calculate silhouette for each sample
        silhouette_vals = []
        
        for i in range(len(X)):
            # Points in the same cluster
            a_cluster = self.labels[i]
            a_indices = np.where(self.labels == a_cluster)[0]
            
            if len(a_indices) > 1:  # Ensure the cluster has more than one point
                # Mean distance to points in the same cluster
                a_i = np.mean([pairwise_distances[i, idx] for idx in a_indices if idx != i])
                
                # Mean distance to points in the nearest cluster
                b_i = float('inf')
                for cluster in range(self.n_clusters):
                    if cluster != a_cluster:
                        b_indices = np.where(self.labels == cluster)[0]
                        if len(b_indices) > 0:
                            cluster_dist = np.mean([pairwise_distances[i, idx] for idx in b_indices])
                            if cluster_dist < b_i:
                                b_i = cluster_dist
                
                # Calculate silhouette
                s_i = (b_i - a_i) / max(a_i, b_i)
                silhouette_vals.append(s_i)
                
        # Return mean silhouette score
        if silhouette_vals:
            return np.mean(silhouette_vals)
        else:
            return 0.0
    
    def plot_clusters(self, X, feature_names=None, figsize=(12, 10)):
        """
        Visualize the clusters in 2D using the first two features or PCA.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The data to visualize.
            
        feature_names : list of str, optional
            Names of the features for axis labels.
            
        figsize : tuple, default=(12, 10)
            Figure size.
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure object.
        """
        if self.centroids is None:
            raise ValueError("Model not trained yet. Call fit() first.")
            
        # Get cluster labels if not already assigned
        if self.labels is None:
            self.labels = self.predict(X)
            
        # Create a figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot data points colored by cluster
        scatter = ax.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', 
                           alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        
        # Plot centroids
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', 
                 marker='X', s=200, label='Centroids', edgecolors='k', linewidths=1.5)
        
        # Set labels and title
        if feature_names and len(feature_names) >= 2:
            ax.set_xlabel(feature_names[0])
            ax.set_ylabel(feature_names[1])
        else:
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
        ax.set_title(f'KMeans Clustering with {self.n_clusters} Clusters')
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def plot_elbow_curve(self, X, k_range=range(1, 11), figsize=(10, 6)):
        """
        Plot the elbow curve to help determine the optimal number of clusters.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            The data to analyze.
            
        k_range : range or list, default=range(1, 11)
            Range of k values to test.
            
        figsize : tuple, default=(10, 6)
            Figure size.
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure object.
        """
        inertias = []
        
        for k in k_range:
            # Create and fit KMeans with k clusters
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            
            # Store inertia
            inertias.append(kmeans.inertia)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot elbow curve
        ax.plot(k_range, inertias, 'o-', color='blue', linewidth=2)
        
        # Add labels and title
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create three distinct clusters
    n_samples = 500
    cluster1 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, -2])
    cluster3 = np.random.randn(n_samples, 2) * 0.5 + np.array([0, 0])
    
    # Combine data
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Print results
    print("Centroids:")
    print(kmeans.centroids)
    print("\nInertia:", kmeans.inertia)
    
    # Plot clusters
    kmeans.plot_clusters(X)
    plt.show()
    
    # Plot elbow curve
    kmeans.plot_elbow_curve(X)
    plt.show()