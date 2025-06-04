# Machine Learning Models for OncoPredictAI

## Overview

The OncoPredictAI system applies advanced machine learning techniques to global cancer datasets for predicting patient outcomes and optimizing treatment strategies. These models are specifically optimized for:

1. **Cancer Risk Stratification**: Identifying high-risk patients based on genetic, environmental, and lifestyle factors
2. **Treatment Outcome Prediction**: Forecasting survival years and severity scores based on patient characteristics
3. **Resource Optimization**: Maximizing healthcare resource allocation for cancer treatment
4. **Interpretable Insights**: Providing clear explanations of predictions for healthcare professionals
5. **Cross-Cultural Application**: Models that can be adapted for use in various healthcare systems, including resource-constrained environments

## Dataset Information

### 1. Global Cancer Patients Dataset

The models in this project can be trained and evaluated using the Global Cancer Patients dataset (2015-2024), which provides comprehensive cancer statistics that can be adapted for the Cambodian healthcare context:

- **Dataset Source**: [Global Cancer Patients Dataset on Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024?resource=download)
- **Contents**: Contains global cancer statistics including patient demographics, cancer types, treatment outcomes, and temporal trends
- **Application**: This dataset can be used to train the clustering, PCA, and predictive models in this project
- **Location**: Download the CSV files from the Kaggle link and place them in the `data/` directory of this project
- **Recommended Algorithms**:
  - K-means Clustering: For patient segmentation and risk factor analysis
  - PCA: For dimensionality reduction and identifying key contributing factors
  - XGBoost/Random Forest: For predicting severity scores and treatment outcomes
  - Light GBM: For resource utilization optimization

### 2. Chest X-ray Pneumonia Dataset

For medical imaging analysis in our system, we use the Chest X-ray Pneumonia dataset:

- **Dataset Source**: [Chest X-ray Pneumonia Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Contents**: Contains chest X-ray images categorized into normal and pneumonia cases
- **Application**: This dataset can be used to train computer vision models for diagnostic support
- **Location**: Download the dataset using the provided script and store in the `data/` directory
- **Recommended Algorithms**:
  - Convolutional Neural Networks (CNNs): For image classification
  - Transfer Learning with pre-trained models: To maximize performance with limited training data

### Automatic Dataset Download

To automatically download the datasets using the kagglehub library:

```python
# Install the required package
pip install kagglehub

# Use the provided download script
python data/download_dataset.py
```

Alternatively, use the kagglehub library directly in your code:

```python
import kagglehub

# Download latest version of the Global Cancer Patients dataset
path = kagglehub.dataset_download("zahidmughal2343/global-cancer-patients-2015-2024")

# Download latest version of the Chest X-ray Pneumonia dataset
path_xray = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to cancer dataset files:", path)
print("Path to X-ray dataset files:", path_xray)
```
## Current ML Implementation

The ml_library currently includes the following custom implementations:

### 1. K-means Clustering (`kmeans.py`)
- **Purpose**: Patient segmentation, resource utilization patterns, and facility zoning
- **Features**:
  - Custom K-means++ initialization for better convergence
  - Memory-efficient implementation for limited hardware resources
  - Interpretable visualization tools for healthcare staff
  - Silhouette score analysis for optimal cluster determination
- **Applications**:
  - Identifying patient groups with similar healthcare needs
  - Detecting patterns in resource usage across different hospital departments
  - Optimizing facility layout based on patient flow data
  - Clustering similar health centers for targeted resource allocation

### 2. Principal Component Analysis (`pca.py`)
- **Purpose**: Dimensionality reduction and feature extraction from healthcare data
- **Features**:
  - Customized for efficient operation on limited hardware
  - Visualization tools specifically designed for healthcare staff interpretability
  - Explained variance analysis for identifying key factors in healthcare operations
- **Applications**:
  - Reducing dimensionality of complex patient data
  - Identifying key factors affecting hospital operations
  - Preprocessing step for other ML models to improve performance
  - Visualizing high-dimensional healthcare data for administrative decision-making

## Planned ML Models

Based on the project requirements, the following additional models will be implemented:

### 3. Time Series Forecasting (ARIMA, Prophet)
- **Purpose**: Patient volume prediction
- **Why Selected**: Captures seasonal disease patterns specific to Cambodia (dengue, malaria, respiratory)
- **Implementation Plan**: Custom wrapper with simplified parameters for non-technical users

### 4. Light GBM
- **Purpose**: Resource utilization optimization
- **Why Selected**: Efficient performance on limited computing hardware available in Cambodian hospitals
- **Implementation Plan**: Optimized version with scaled-down default parameters for faster execution

### 5. Random Forest
- **Purpose**: Triage optimization
- **Why Selected**: Robust to noisy data common in transitioning systems; interpretable for staff with limited AI exposure
- **Implementation Plan**: Feature importance visualization focused on actionable insights

### 6. XGBoost
- **Purpose**: Length of stay prediction
- **Why Selected**: Superior performance with limited training data; handles class imbalance common in developing settings
- **Implementation Plan**: Ensemble approach with customized early stopping for resource efficiency

### 7. Ensemble Methods
- **Purpose**: Disease outbreak detection
- **Why Selected**: Combines multiple data sources to provide early warnings for Cambodia's specific endemic diseases
- **Implementation Plan**: Weighted voting mechanism with special focus on Cambodia's disease patterns

### 8. Federated Learning
- **Purpose**: Multi-facility modeling
- **Why Selected**: Allows learning across hospitals without centralizing sensitive data, respecting privacy with limited regulatory infrastructure
- **Implementation Plan**: Simplified implementation focused on core privacy-preserving features

## Implementation Priorities

The implementation roadmap prioritizes models that:

1. Deliver immediate operational value to Cambodian healthcare facilities
2. Can function effectively with limited initial data
3. Operate efficiently on available hardware infrastructure
4. Provide interpretable results for building trust in the system

## Model Evaluation Approach

Models will be evaluated based on:

1. **Performance Metrics**: Accuracy, precision, recall, F1-score
2. **Resource Utilization**: Memory usage, inference time, CPU load
3. **Interpretability**: Clarity of outputs for healthcare administrators
4. **Resilience**: Performance with missing or noisy data
5. **Real-world Impact**: Measured improvements in hospital operations

## Custom Hyperparameters

Hyperparameters are specifically optimized for Cambodia's healthcare context:

```python
# Example hyperparameters for resource allocation optimization
lgbm_params = {
    'max_depth': 6,                  # Limited depth for faster inference
    'learning_rate': 0.05,           # Balanced for learning speed and stability
    'n_estimators': 200,             # Optimized for limited computing resources
    'subsample': 0.8,                # Provides robustness to noisy data
    'colsample_bytree': 0.75,        # Feature sampling for better generalization
    'min_child_weight': 3,           # Prevents overfitting on limited data
    'objective': 'binary:logistic',
    'device': 'cpu',                 # Optimized for CPU-only environments
    'verbose': -1
}
```

## Integration Strategy

These ML models are integrated into the CamCare system through:

1. **API Layer**: Standardized interfaces for model training and inference
2. **Caching Mechanisms**: Results storage for offline operation
3. **Scheduled Retraining**: Automatic model updates as new data becomes available
4. **Explainability Tools**: Visualizations and simplified explanations of model decisions
5. **Feedback Loops**: Systems for healthcare staff to provide feedback on model performance

## Future Enhancements

As the digital transformation of Cambodia's healthcare system progresses, the ML strategy will evolve to include:

1. **Advanced NLP**: For processing clinical notes in both Khmer and English
2. **Computer Vision**: For medical imaging analysis where appropriate
3. **Reinforcement Learning**: For optimizing complex hospital workflows
4. **Transfer Learning**: Leveraging global healthcare models adapted to Cambodia's specific context

## Training and Support

To ensure effective utilization of these ML models:

1. **Tiered Training**: Different depth of ML understanding for technical vs. clinical staff
2. **Visual Dashboards**: Intuitive interfaces for interacting with model outputs
3. **Decision Support Documentation**: Clear guidelines on how to interpret and act on model predictions
4. **Local Expertise Development**: Programs to train Cambodian staff in ML maintenance and enhancement
