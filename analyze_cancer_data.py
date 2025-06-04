"""
Cancer Data Analysis Starter Script for OncoPredictAI

This script provides a starting point for analyzing the Global Cancer Patients dataset
using the existing PCA and K-means implementations.

Usage:
    python analyze_cancer_data.py

Requirements:
    - pandas, numpy, matplotlib, seaborn
    - Models from the project structure (dimensionality_reduction/pca.py, clustering/kmeans.py)
    - Configuration from config/default.yaml
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from models.dimensionality_reduction.pca import PCA
from models.clustering.kmeans import KMeans
from src.data.preprocessing import preprocess_data, handle_outliers
from src.visualization.visualize import set_visualization_style, plot_correlation_matrix, plot_pca_variance, plot_cluster_distribution, save_figure

# Add the project directory to the path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

# Load configuration
config_path = os.path.join(project_dir, "config", "default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Set visualization style based on configuration
viz_config = config.get('visualization', {})
style = viz_config.get('style', 'whitegrid')
context = viz_config.get('context', 'notebook')
palette = viz_config.get('palette', 'viridis')
set_visualization_style(style, context, palette)

def load_cancer_data():
    """
    Load the global cancer patients dataset
    """
    # Get data path from config
    data_path = config.get('data', {}).get('raw_cancer_data', 
        os.path.join("data", "raw", "cancer_patients", "global_cancer_patients_2015_2024.csv"))
    
    # Try loading from the config path first
    if not os.path.exists(data_path):
        # Fall back to old location if file not found
        data_path = os.path.join("data", "global_cancer_patients_2015_2024.csv")
        
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}")
            print("Please run the download script first: python scripts/download_dataset.py")
            sys.exit(1)
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded successfully with {len(df)} rows and {df.shape[1]} columns")
    return df

def explore_data(df):
    """
    Perform initial exploratory data analysis
    """
    print("\n=== Dataset Overview ===")
    print(df.head())
    
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() > 0 else "No missing values found")
    
    print("\n=== Summary Statistics ===")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    print(df[numeric_cols].describe())
    
    # Cancer type distribution
    print("\n=== Cancer Type Distribution ===")
    cancer_counts = df["Cancer_Type"].value_counts()
    print(cancer_counts)
    
    # Create a figure with multiple plots
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Age distribution
    plt.subplot(2, 3, 1)
    sns.histplot(df["Age"], kde=True)
    plt.title("Age Distribution")
    
    # Plot 2: Cancer Types
    plt.subplot(2, 3, 2)
    sns.countplot(y="Cancer_Type", data=df, order=df["Cancer_Type"].value_counts().index)
    plt.title("Cancer Types")
    
    # Plot 3: Cancer Stage
    plt.subplot(2, 3, 3)
    sns.countplot(y="Cancer_Stage", data=df, order=df["Cancer_Stage"].value_counts().index)
    plt.title("Cancer Stages")
    
    # Plot 4: Risk factor correlation heatmap
    plt.subplot(2, 3, 4)
    risk_factors = ["Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
    sns.heatmap(df[risk_factors].corr(), annot=True, cmap="coolwarm")
    plt.title("Risk Factor Correlations")
    
    # Plot 5: Survival Years vs Severity Score
    plt.subplot(2, 3, 5)
    sns.scatterplot(x="Target_Severity_Score", y="Survival_Years", data=df, alpha=0.5)
    plt.title("Survival Years vs Severity Score")
    
    # Plot 6: Age vs Treatment Cost
    plt.subplot(2, 3, 6)
    sns.boxplot(x="Cancer_Stage", y="Treatment_Cost_USD", data=df)
    plt.title("Treatment Cost by Cancer Stage")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("data/cancer_data_overview.png")
    print("\nEDA visualizations saved to data/cancer_data_overview.png")
    
def apply_pca(df):
    """
    Apply PCA to the cancer dataset to reduce dimensionality
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        tuple: (X_pca, pca) - PCA-transformed data and PCA model
    """
    print("\n=== Applying PCA ===")
    
    # Select numerical features and remove target variables
    target_cols = ['Target_Severity_Score', 'Survival_Years']
    feature_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in target_cols]
    
    # Extract features for PCA
    X = df[feature_cols].values
    
    # Get number of components from config or use default
    n_components = config.get('models', {}).get('pca', {}).get('n_components', 2)
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Print explained variance
    if hasattr(pca, 'explained_variance_ratio'):
        explained_variance = pca.explained_variance_ratio
        print(f"Explained variance ratio: {explained_variance}")
        total_variance = np.sum(explained_variance) if explained_variance is not None else 0
        print(f"Total explained variance: {total_variance:.4f}")
      # Plot explained variance
    try:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        if explained_variance is not None:
            components = np.arange(1, len(explained_variance) + 1)
            ax.bar(components, explained_variance)
            ax.plot(components, np.cumsum(explained_variance), 'r-o')
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig("outputs/figures/cancer_data_pca.png")
            print("PCA visualization saved to outputs/figures/cancer_data_pca.png")
        else:
            print("No explained variance data available for plotting")
    except Exception as e:
        print(f"Error plotting PCA variance: {str(e)}")
    
    # Plot 2D projection if n_components >= 2
    if n_components >= 2:
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            # Create scatter plot of first two components
            if X_pca is not None and X_pca.shape[1] >= 2:
                # Add color based on cancer type if available
                if 'Cancer_Type' in df.columns:
                    cancer_types = df['Cancer_Type'].unique()
                    for i, cancer_type in enumerate(cancer_types):
                        idx = df['Cancer_Type'] == cancer_type
                        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=cancer_type)
                    ax.legend()
                else:
                    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
                    
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title("PCA 2D Projection of Cancer Data")
                ax.grid(True)
                plt.tight_layout()
                plt.savefig("outputs/figures/cancer_data_pca_2d.png")
                print("PCA 2D projection saved to outputs/figures/cancer_data_pca_2d.png")
        except Exception as e:
            print(f"Error plotting PCA 2D projection: {str(e)}")
    
    return X_pca, pca

def apply_kmeans(X_normalized, df):
    """
    Apply KMeans clustering to the normalized data
    
    Args:
        X_normalized: Normalized or PCA-transformed data
        df: Original DataFrame for reference
    
    Returns:
        tuple: (labels, kmeans) - Cluster labels and KMeans model
    """
    print("\n=== Applying K-means Clustering ===")
    
    # Get clustering parameters from config
    kmeans_config = config.get('models', {}).get('kmeans', {})
    n_clusters = kmeans_config.get('n_clusters', 3)
    random_state = kmeans_config.get('random_state', 42)
    max_iter = kmeans_config.get('max_iter', 300)
    
    # Initialize K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iters=max_iter
    )
    
    # Fit and predict clusters
    labels = kmeans.fit_predict(X_normalized)
    
    # Print clustering results
    print(f"K-means clustering completed with {n_clusters} clusters")
    if hasattr(kmeans, 'inertia'):
        print(f"Inertia: {kmeans.inertia}")
    
    # Plot clusters
    plt.figure(figsize=(12, 10))
    kmeans.plot_clusters(X_normalized)
    plt.title(f"K-means Clustering of Cancer Data ({n_clusters} clusters)")
    plt.savefig("outputs/figures/kmeans_clusters.png")
    print("Clustering results saved to outputs/figures/kmeans_clusters.png")
    
    # Plot elbow curve to find optimal k
    plt.figure(figsize=(10, 6))
    kmeans.plot_elbow_curve(X_normalized, k_range=range(1, 11))
    plt.title("Elbow Curve for Optimal K Selection")
    plt.savefig("outputs/figures/kmeans_elbow.png")
    print("Elbow curve saved to outputs/figures/kmeans_elbow.png")
    
    # Plot cluster distribution
    plt.figure(figsize=(10, 6))
    plot_cluster_distribution(labels)
    plt.savefig("outputs/figures/cluster_distribution.png")
    print("Cluster distribution saved to outputs/figures/cluster_distribution.png")
    
    # Add cluster labels to the original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Analyze clusters
    print("\n=== Cluster Analysis ===")
    for cluster_id in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} samples):")
        
        # Show most common cancer types and stages in this cluster
        if 'Cancer_Type' in df.columns:
            print("  Top Cancer Types:")
            print(cluster_data['Cancer_Type'].value_counts().head(3))
            
        if 'Cancer_Stage' in df.columns:
            print("  Cancer Stage Distribution:")
            print(cluster_data['Cancer_Stage'].value_counts())
        
        # Show average age, risk factors, survival years
        numeric_cols = ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 
                        'Smoking', 'Obesity_Level', 'Target_Severity_Score', 
                        'Survival_Years', 'Treatment_Cost_USD']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if numeric_cols:
            print("  Key Metrics (Mean):")
            for col in numeric_cols:
                print(f"    {col}: {cluster_data[col].mean():.2f}")
    
    return labels, kmeans

if __name__ == "__main__":
    # Load the dataset
    cancer_df = load_cancer_data()
    
    # Explore the data
    explore_data(cancer_df)
      # Apply PCA
    X_norm, pca_model = apply_pca(cancer_df)
    
    # Add PCA results to the main dataframe
    # X_norm contains the transformed data with columns for each principal component
    cancer_df['PC1'] = X_norm[:, 0]  # First principal component
    cancer_df['PC2'] = X_norm[:, 1]  # Second principal component
    
    # Apply K-means clustering
    clustered_df = apply_kmeans(X_norm, cancer_df)
    
    print("\nAnalysis complete! Check the data/ directory for visualization outputs.")
