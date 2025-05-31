# Healthcare ML Model Training Guide

## 1. Data Preparation

### Data Collection
1. **Patient Records**
   - Demographics
   - Medical history
   - Lab results
   - Medications
   - Treatment outcomes
   - Recovery times

2. **Data Sources Integration**
   ```python
   # Example code structure for data integration
   def load_data():
       patient_data = pd.read_csv('patient_records.csv')
       lab_results = pd.read_csv('lab_results.csv')
       medications = pd.read_csv('medications.csv')
       return merge_datasets(patient_data, lab_results, medications)
   ```

### Data Preprocessing
1. **Handling Missing Values**
   ```python
   def handle_missing_values(df):
       # Numerical columns: fill with median
       numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
       df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
       
       # Categorical columns: fill with mode
       categorical_cols = df.select_dtypes(include=['object']).columns
       df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
       return df
   ```

2. **Feature Scaling**
   ```python
   from sklearn.preprocessing import StandardScaler, MinMaxScaler
   
   def scale_features(X_train, X_test):
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)
       return X_train_scaled, X_test_scaled
   ```

## 2. Feature Engineering

### Creating Features
1. **Time-based Features**
   ```python
   def create_time_features(df):
       df['age'] = calculate_age(df['date_of_birth'])
       df['treatment_duration'] = calculate_duration(df['start_date'], df['end_date'])
       df['previous_admissions'] = calculate_previous_admissions(df)
       return df
   ```

2. **Medical History Features**
   ```python
   def create_medical_features(df):
       df['chronic_condition_count'] = calculate_conditions(df['medical_history'])
       df['medication_interactions'] = analyze_drug_interactions(df['medications'])
       return df
   ```

## 3. Model Development

### Base Models
1. **Random Forest for Disease Prediction**
   ```python
   def train_random_forest(X_train, y_train):
       rf_params = {
           'n_estimators': 500,
           'max_depth': 20,
           'min_samples_split': 5,
           'random_state': 42
       }
       rf_model = RandomForestClassifier(**rf_params)
       rf_model.fit(X_train, y_train)
       return rf_model
   ```

2. **XGBoost for Recovery Time Prediction**
   ```python
   def train_xgboost(X_train, y_train):
       xgb_params = {
           'learning_rate': 0.1,
           'max_depth': 6,
           'n_estimators': 200,
           'objective': 'reg:squarederror'
       }
       xgb_model = XGBRegressor(**xgb_params)
       xgb_model.fit(X_train, y_train)
       return xgb_model
   ```

## 4. Model Training Pipeline

### Training Workflow
```python
def train_models_pipeline(data_path):
    # 1. Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    
    # 2. Feature engineering
    df = create_features(df)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1),
        df['target'],
        test_size=0.2,
        random_state=42
    )
    
    # 4. Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # 5. Train models
    models = {}
    models['random_forest'] = train_random_forest(X_train_scaled, y_train)
    models['xgboost'] = train_xgboost(X_train_scaled, y_train)
    
    # 6. Evaluate models
    evaluate_models(models, X_test_scaled, y_test)
    
    return models
```

## 5. Hyperparameter Tuning

### Grid Search
```python
def tune_hyperparameters(model, X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
```

## 6. Model Evaluation

### Metrics and Validation
```python
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
```

## 7. Model Validation

### Cross-Validation
```python
def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 8. Production Deployment

### Model Saving
```python
def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save model metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'model_version': '1.0',
        'features': model.feature_names_,
        'parameters': model.get_params()
    }
    
    with open(f"{model_path}_metadata.json", 'w') as f:
        json.dump(metadata, f)
```

## Best Practices

1. **Data Quality**
   - Always validate data quality before training
   - Handle missing values appropriately
   - Check for data imbalance

2. **Model Selection**
   - Start with simple models
   - Gradually increase complexity
   - Use ensemble methods for robust predictions

3. **Validation Strategy**
   - Use stratified k-fold cross-validation
   - Maintain a separate test set
   - Monitor for overfitting

4. **Performance Monitoring**
   - Track model drift
   - Monitor prediction quality
   - Set up automated retraining

5. **Documentation**
   - Document all preprocessing steps
   - Record hyperparameter choices
   - Maintain model versions

## Regular Maintenance

1. **Model Retraining Schedule**
   - Retrain models monthly
   - Update feature engineering as needed
   - Validate on new data

2. **Performance Checks**
   - Monitor prediction accuracy
   - Check for data drift
   - Validate against baseline models

3. **Update Procedures**
   - Version control for models
   - Backup previous versions
   - Document changes and improvements
