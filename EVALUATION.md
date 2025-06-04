# Model Evaluation Strategy

## Metrics Used
- **Clustering:** Silhouette score, inertia
- **Regression:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared
- **Classification:** Accuracy, precision, recall, F1-score

## Evaluation Approach
- Compare models using cross-validation and holdout sets
- Assess resource utilization (memory, CPU, inference time)
- Emphasize interpretability and resilience to missing/noisy data
- Measure real-world impact on hospital operations

## Tools
- Evaluation utilities in `src/utils/evaluation.py`
- Visualizations in `src/visualization/visualize.py`
