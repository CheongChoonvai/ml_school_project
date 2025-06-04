# OncoPredictAI: User Guide

This guide explains how to set up and run the OncoPredictAI cancer prediction and analysis system. It provides step-by-step instructions for getting started with the project, downloading required datasets, and running various analysis and prediction tasks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Data Acquisition](#data-acquisition)
5. [Running Basic Analysis](#running-basic-analysis)
6. [Running ML Pipelines](#running-ml-pipelines)
7. [Available Models](#available-models)
8. [Visualization and Results](#visualization-and-results)
9. [Troubleshooting](#troubleshooting)

## Project Overview

OncoPredictAI is a machine learning framework designed for cancer data analysis and prediction. The system can:
- Analyze cancer patient data to identify patterns and trends
- Perform clustering to identify patient groups with similar characteristics
- Apply dimensionality reduction to visualize complex cancer datasets
- Build predictive models for cancer severity and outcomes
- Generate visualizations and insights for healthcare professionals

## Prerequisites

Before getting started, ensure you have the following:

- Python 3.10 or higher
- Git (for cloning the repository)
- Kaggle account (for dataset access)
- Sufficient disk space for datasets (approximately 2GB)

## Environment Setup

You can set up the environment using either conda or pip:

### Option 1: Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate oncopredict-ai
```

### Option 2: Using Pip

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Data Acquisition

Before running any analysis, you need to download the required datasets:

```bash
# Run the dataset download script
python scripts\download_dataset.py
```

The script will prompt you to choose which datasets to download:
1. Global Cancer Patients dataset (2015-2024)
2. Chest X-ray Pneumonia dataset
3. Both datasets

Select option 1 or 3 to download the cancer dataset, which is required for most analyses.

The script uses the Kaggle API, so you'll need a Kaggle account and API credentials:
1. Create a Kaggle account at https://www.kaggle.com (if you don't have one)
2. Go to 'Account' > 'Create API Token' to download kaggle.json
3. Place kaggle.json in `%USERPROFILE%\.kaggle\` on Windows or `~/.kaggle/` on Linux/Mac
4. Ensure permissions are set correctly (chmod 600 ~/.kaggle/kaggle.json on Linux/Mac)

## Running Basic Analysis

For a quick start with exploratory analysis of the cancer dataset, run:

```bash
# Run the basic analysis script
python analyze_cancer_data.py
```

This script will:
- Load the cancer dataset
- Perform basic exploratory data analysis
- Generate visualizations for key insights
- Run PCA and K-means clustering on the dataset
- Save visualization outputs to the data/ directory

## Running ML Pipelines

For more advanced machine learning pipelines, use the main entry point:

```bash
# Basic usage
python src\main.py --config config\default.yaml --model random_forest --mode train
```

### Command-line Arguments:

- `--config`: Path to the configuration file (default: config/default.yaml)
- `--model`: Machine learning model to use (choices: kmeans, pca, random_forest, xgboost)
- `--mode`: Operation mode (choices: train, predict, evaluate, default: train)

### Example Commands:

Train a Random Forest classifier:
```bash
python src\main.py --model random_forest --mode train
```

Evaluate a trained XGBoost model:
```bash
python src\main.py --model xgboost --mode evaluate
```

Make predictions with a K-means clustering model:
```bash
python src\main.py --model kmeans --mode predict
```

## Available Models

The project currently supports the following models:

1. **K-means Clustering (kmeans)**
   - Patient segmentation and grouping
   - Detecting patterns in resource usage
   - Identifying similar cancer cases

2. **Principal Component Analysis (pca)**
   - Dimensionality reduction
   - Feature importance analysis
   - Data visualization

3. **Random Forest (random_forest)**
   - Classification of cancer types
   - Prediction of cancer severity
   - Interpretable feature importance

4. **XGBoost (xgboost)**
   - High-performance prediction
   - Handling imbalanced cancer data
   - Survival prediction

## Visualization and Results

After running the models, visualizations and results are saved in the following directories:

- `outputs/figures/`: Contains all generated plots and visualizations
- `outputs/models/`: Saved trained model files
- `outputs/results/`: Model evaluation and prediction results

Key visualizations include:
- PCA component analysis plots
- K-means clustering visualizations
- Feature importance charts
- Confusion matrices for classification
- Survival prediction curves

## Troubleshooting

### Common Issues:

1. **Dataset not found errors**
   - Make sure you've run `scripts\download_dataset.py` first
   - Verify that the paths in `config\default.yaml` match your file locations

2. **Module import errors**
   - Ensure your environment is activated
   - Check that you're running commands from the project root directory

3. **Memory errors when processing large datasets**
   - Reduce batch sizes in the configuration
   - Use a subset of the data for initial exploration

4. **Kaggle API authentication errors**
   - Verify your kaggle.json file is correctly placed
   - Check permissions on the kaggle.json file

For any other issues, please refer to the project documentation or submit an issue on the project repository.

---

This user guide provides basic instructions for getting started with OncoPredictAI. For more detailed information about the project structure and workflow, please refer to the `PROJECT_WORKFLOW.md` file.
