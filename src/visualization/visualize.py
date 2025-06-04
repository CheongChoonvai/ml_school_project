"""
Visualization utilities for OncoPredictAI

This module provides functions for creating various visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def set_visualization_style(style='whitegrid', context='notebook', palette='viridis'):
    """
    Set the visualization style for all plots.
    
    Args:
        style: Seaborn style
        context: Seaborn context
        palette: Color palette
    """
    sns.set_style(style)
    sns.set_context(context)
    sns.set_palette(palette)
    
    logger.info(f"Visualization style set to: {style}, context: {context}, palette: {palette}")

def plot_feature_importance(model, feature_names, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    logger.info(f"Plotting feature importance")
    
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Take top N features
    if top_n is not None:
        feature_imp = feature_imp.head(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {len(feature_imp)} Feature Importances')
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, 'feature_importance')
    
    return fig

def plot_correlation_matrix(df, method='pearson', figsize=(14, 12), mask_upper=True):
    """
    Plot correlation matrix for DataFrame.
    
    Args:
        df: DataFrame
        method: Correlation method ('pearson', 'kendall', 'spearman')
        figsize: Figure size
        mask_upper: Whether to mask the upper triangle
        
    Returns:
        matplotlib figure
    """
    logger.info(f"Plotting correlation matrix")
    
    # Calculate correlation matrix
    corr = df.corr(method=method)
    
    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = sns.heatmap(
        corr, 
        mask=mask,
        vmin=-1, vmax=1, 
        annot=True, 
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax
    )
    
    # Add title
    ax.set_title(f'Feature Correlation Matrix ({method.capitalize()})')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, 'correlation_matrix')
    
    return fig

def plot_pca_variance(pca_model, figsize=(10, 6)):
    """
    Plot explained variance from PCA model.
    
    Args:
        pca_model: Trained PCA model with explained_variance_ratio_ attribute
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    logger.info(f"Plotting PCA explained variance")
    
    # Check if model has explained_variance_ratio_
    if not hasattr(pca_model, 'explained_variance_ratio_'):
        logger.warning("PCA model does not have explained_variance_ratio_ attribute")
        return None
    
    # Get explained variance
    explained_variance = pca_model.explained_variance_ratio_
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual explained variance
    ax.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
        alpha=0.7,
        label='Individual'
    )
    
    # Plot cumulative explained variance
    ax.plot(
        range(1, len(explained_variance) + 1),
        np.cumsum(explained_variance),
        'r-o',
        alpha=0.7,
        label='Cumulative'
    )
    
    # Add horizontal line at 0.95
    ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label='95% Threshold')
    
    # Add labels and title
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, 'pca_variance')
    
    return fig

def plot_cluster_distribution(labels, figsize=(10, 6)):
    """
    Plot distribution of cluster labels.
    
    Args:
        labels: Cluster labels
        figsize: Figure size
        
    Returns:
        matplotlib figure
    """
    logger.info(f"Plotting cluster distribution")
    
    # Count clusters
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    cluster_counts.plot(kind='bar', ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('Cluster Distribution')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add count labels on top of bars
    for i, count in enumerate(cluster_counts):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, 'cluster_distribution')
    
    return fig

def save_figure(fig, name, dpi=300):
    """
    Save figure to disk.
    
    Args:
        fig: Matplotlib figure
        name: Base name for the file
        dpi: Resolution
        
    Returns:
        str: Path to saved figure
    """
    import os
    
    # Create directory if it doesn't exist
    fig_dir = os.path.join('outputs', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Generate filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(fig_dir, filename)
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    
    logger.info(f"Figure saved to {filepath}")
    
    return filepath
