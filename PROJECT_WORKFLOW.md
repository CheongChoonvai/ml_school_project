# OncoPredictAI Project Workflow

## Overview

OncoPredictAI is a machine learning framework designed for cancer data analysis and prediction. This document outlines the workflow and architecture of the project, explaining how different components interact and how data flows through the system.

## Data Flow Architecture

```
[Data Sources] → [Data Acquisition] → [Data Preprocessing] → [Feature Engineering] 
→ [Model Training] → [Evaluation] → [Visualization] → [Clinical Application]
```

## Components and Workflow

### 1. Data Acquisition

**Scripts**: `scripts/download_dataset.py`

This component is responsible for acquiring cancer datasets from external sources, primarily from Kaggle:

1. **Global Cancer Patients Dataset (2015-2024)**: Contains comprehensive cancer statistics including patient demographics, cancer types, treatment outcomes, and temporal trends.
2. **Chest X-ray Pneumonia Dataset**: Contains chest X-ray images for medical imaging analysis.

The data acquisition process:
1. Uses the `kagglehub` library to access and download datasets
2. Verifies dataset integrity and provides basic statistics
3. Stores datasets in the `data/raw/` directory for further processing

### 2. Data Loading and Preprocessing

**Modules**: `src/data/make_dataset.py`, `src/data/preprocessing.py`

Once acquired, the data undergoes preprocessing to prepare it for analysis:

1. **Data Loading**: Handles various file formats (CSV, Excel, JSON) and validates data existence
2. **Data Cleaning**:
   - Identifies and handles missing values using appropriate imputation strategies
   - Removes outliers that could skew model results
3. **Feature Transformation**:
   - Standardizes numerical features
   - Encodes categorical variables using one-hot encoding or similar techniques
4. **Dataset Splitting**: Divides data into training, validation, and testing sets

### 3. Exploratory Data Analysis

**Notebooks**: `notebooks/exploration/01_cancer_data_exploration.ipynb`

Before building models, the data is thoroughly explored to gain insights:

1. **Statistical Summaries**: Generate descriptive statistics for all features
2. **Distribution Analysis**: Visualize the distributions of key variables:
   - Age distribution
   - Risk factors (Genetic_Risk, Air_Pollution, Alcohol_Use, Smoking, Obesity_Level)
   - Cancer types and stages
   - Treatment costs and survival years
3. **Correlation Analysis**: Identify relationships between variables
4. **Target Variable Analysis**: Understand the distribution and characteristics of prediction targets

### 4. Feature Engineering

**Modules**: `src/features/build_features.py`

This component transforms raw data into features suitable for machine learning:

1. **Feature Selection**: Identifies most relevant features for prediction
2. **Feature Creation**: Derives new features from existing ones
3. **Dimensionality Reduction**: Uses PCA to reduce data dimensions while preserving information
4. **Feature Scaling**: Ensures all features are on similar scales for algorithm performance

### 5. Model Development

**Modules**: Various models in `models/` directory, `src/models/train_model.py`

The system implements several machine learning approaches:

1. **Clustering Models**:
   - K-means clustering for patient segmentation (`models/clustering/kmeans.py`)
   - Used to identify distinct patient groups based on similar characteristics

2. **Dimensionality Reduction**:
   - PCA implementation for feature analysis (`models/dimensionality_reduction/pca.py`)
   - Used to reduce data complexity and visualize high-dimensional data

3. **Predictive Models**:
   - Random Forest for classification and regression tasks
   - XGBoost for high-performance prediction
   - Used to predict severity scores and survival years

### 6. Model Training and Evaluation

**Modules**: `src/models/train_model.py`, `src/utils/evaluation.py`

The model training process:

1. **Training Pipeline**:
   - Configurable through YAML files in `config/`
   - Supports different algorithms and hyperparameter configurations
   - Handles logging of training progress and results

2. **Evaluation Metrics**:
   - For clustering: silhouette score, inertia
   - For regression: MAE, RMSE, R-squared
   - For classification: accuracy, precision, recall, F1-score

### 7. Visualization

**Modules**: `src/visualization/visualize.py`

Results are visualized for interpretation:

1. **Clustering Visualizations**:
   - 2D/3D representations of patient clusters
   - Characterization of clusters by key attributes

2. **Model Performance Visualizations**:
   - Learning curves
   - Feature importance plots
   - Confusion matrices for classification tasks
   - Prediction vs. actual plots for regression

3. **Data Insight Visualizations**:
   - Distribution plots
   - Correlation heatmaps
   - PCA component analysis

### 8. Main Pipeline Execution

**Scripts**: `src/main.py`

The main entry point orchestrates the entire workflow:

1. Parses command-line arguments for configuration path and model selection
2. Loads configurations from YAML files
3. Executes the appropriate pipeline based on the selected model and mode:
   - `train`: Train a new model
   - `predict`: Make predictions using an existing model
   - `evaluate`: Evaluate model performance

## Project Structure

```
ml-project/
├── config/                # Configuration files
├── data/                  # Data storage
│   ├── raw/               # Original, immutable data
│   └── processed/         # Cleaned, processed data
├── models/                # Model implementations
│   ├── clustering/        # Clustering algorithms
│   ├── dimensionality_reduction/
│   └── classification/    # Predictive models
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── scripts/               # Utility scripts
└── src/                   # Source code
    ├── data/              # Data loading and processing
    ├── features/          # Feature engineering
    ├── models/            # Model training and prediction
    ├── utils/             # Utility functions
    └── visualization/     # Visualization utilities
```

## Execution Flow

1. **Dataset Acquisition**:
   ```bash
   python scripts/download_dataset.py
   ```

2. **Exploratory Analysis**:
   - Run Jupyter notebooks in `notebooks/exploration/`

3. **Model Training**:
   ```bash
   python src/main.py --config config/default.yaml --model kmeans --mode train
   ```

4. **Model Evaluation**:
   ```bash
   python src/main.py --config config/default.yaml --model kmeans --mode evaluate
   ```

5. **Prediction**:
   ```bash
   python src/main.py --config config/default.yaml --model kmeans --mode predict
   ```

## Key Features of OncoPredictAI

1. **Cancer Risk Stratification**: Identifying high-risk patients based on genetic, environmental, and lifestyle factors
2. **Treatment Outcome Prediction**: Forecasting survival years and severity scores based on patient characteristics
3. **Resource Optimization**: Maximizing healthcare resource allocation for cancer treatment
4. **Interpretable Insights**: Providing clear explanations of predictions for healthcare professionals
5. **Cross-Cultural Application**: Models that can be adapted for use in various healthcare systems, including resource-constrained environments

## Conclusion

The OncoPredictAI workflow is designed to provide a comprehensive solution for cancer data analysis and prediction. By following the data-driven approach outlined in this document, healthcare professionals can gain valuable insights into cancer patterns, patient outcomes, and resource optimization strategies.
