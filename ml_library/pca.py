"""
Principal Component Analysis (PCA) Implementation from Scratch
Designed for the CamCare Hospital Management System in Cambodia

This implementation is optimized for computational efficiency in environments
with limited hardware resources, focusing on interpretability for healthcare
staff with varying levels of AI familiarity.
"""

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    """
    Custom PCA implementation designed for Cambodian healthcare data analysis.
    
    This implementation can work effectively with limited computing resources,
    handles noisy data common in transitioning healthcare systems, and
    provides visualizations that are interpretable to healthcare staff.
    """
    
    def __init__(self, n_components=None):
        """
        Initialize PCA with the number of components to keep.
        
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to retain.
            If None, all components are kept.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        """
        Fit the PCA model to the data.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data to fit the PCA model.
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # Mean centering the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort the eigenvalues and corresponding eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        self.explained_variance = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate explained variance ratio
        self.explained_variance_ratio = self.explained_variance / np.sum(self.explained_variance)
        
        # Store the principal components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        # Select the top n_components
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Transform the data to the new basis defined by the principal components.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Data to transform.
            
        Returns:
        --------
        X_transformed : numpy array, shape (n_samples, n_components)
            Transformed data in the principal component space.
        """
        # Check if components exist
        if self.components is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
            
        # Mean center the data
        X_centered = X - self.mean
        
        # Project the centered data onto the principal components
        X_transformed = np.dot(X_centered, self.components)
        
        return X_transformed
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space from the PCA subspace.
        
        Parameters:
        -----------
        X_transformed : numpy array, shape (n_samples, n_components)
            Data in the principal component space.
            
        Returns:
        --------
        X_reconstructed : numpy array, shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        # Check if components exist
        if self.components is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
            
        # Project principal components back to the original space
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean
        
        return X_reconstructed
    
    def fit_transform(self, X):
        """
        Fit the model to the data and apply dimensionality reduction.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_transformed : numpy array, shape (n_samples, n_components)
            Transformed data in the principal component space.
        """
        self.fit(X)
        return self.transform(X)
    
    def plot_explained_variance(self, cumulative=True, figsize=(10, 6)):
        """
        Plot the explained variance by each principal component.
        
        Parameters:
        -----------
        cumulative : bool, default=True
            Whether to plot the cumulative explained variance or individual variance.
        figsize : tuple, default=(10, 6)
            Figure size.
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure object.
        """
        # Check if explained_variance_ratio exists
        if self.explained_variance_ratio is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        components = range(1, len(self.explained_variance_ratio) + 1)
        
        if cumulative:
            # Cumulative explained variance
            cumulative_variance = np.cumsum(self.explained_variance_ratio)
            ax.plot(components, cumulative_variance, marker='o', linestyle='-')
            ax.set_ylabel('Cumulative Explained Variance Ratio')
            ax.set_title('Cumulative Explained Variance by Components')
            
            # Add a horizontal line at 0.95 for reference
            ax.axhline(y=0.95, color='r', linestyle='--', 
                      label='95% Variance Threshold')
            
        else:
            # Individual explained variance
            ax.bar(components, self.explained_variance_ratio)
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Explained Variance by Components')
            
        ax.set_xlabel('Principal Components')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_2d_projection(self, X, y=None, figsize=(10, 8)):
        """
        Plot a 2D projection of the data using the first two principal components.
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Data to transform and plot.
        y : numpy array, shape (n_samples,), optional
            Target values for coloring the points.
        figsize : tuple, default=(10, 8)
            Figure size.
            
        Returns:
        --------
        fig : matplotlib Figure
            The figure object.
        """
        # Check if model has been fitted and has at least 2 components
        if self.components is None:
            raise ValueError("PCA model has not been fitted yet. Call fit() first.")
        if self.components.shape[1] < 2:
            raise ValueError("PCA model needs at least 2 components for 2D projection.")
        if self.explained_variance_ratio is None or len(self.explained_variance_ratio) < 2:
            raise ValueError("Explained variance ratio not available. Call fit() first.")
            
        # Transform the data
        X_transformed = self.transform(X)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # If class labels are provided, color the points accordingly
        if y is not None:
            scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, 
                        alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 ({self.explained_variance_ratio[0]:.2%} variance)')
        ax.set_ylabel(f'Principal Component 2 ({self.explained_variance_ratio[1]:.2%} variance)')
        ax.set_title('2D PCA Projection')
        ax.grid(True)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Create a synthetic dataset with some correlation structure
    X = np.random.randn(n_samples, n_features)
    X[:, 0] = X[:, 1] + X[:, 2] + np.random.randn(n_samples) * 0.5  # Add correlation
    X[:, 3] = X[:, 4] + np.random.randn(n_samples) * 0.3  # Add correlation
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
      # Print the explained variance
    if pca.explained_variance_ratio is not None:
        print("Explained variance ratio:", pca.explained_variance_ratio)
        print("Cumulative explained variance:", np.sum(pca.explained_variance_ratio[:2]))
    
    # Plot the explained variance
    pca.plot_explained_variance()
    plt.show()
    
    # Plot the 2D projection
    pca.plot_2d_projection(X)
    plt.show()
    
    # Reconstruct the data
    X_reconstructed = pca.inverse_transform(X_transformed)
    print("Reconstruction error:", np.mean((X - X_reconstructed) ** 2))
