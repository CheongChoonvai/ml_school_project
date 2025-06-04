oncology_ai/                       # Root directory with a clear project name
├── config/                        # Configuration files
│   ├── default.yaml               # Default configuration
│   ├── experiment_configs/        # Specific experiment configurations
│   └── hyperparameters/           # Hyperparameter configurations for different models
│
├── data/                          # All data-related files
│   ├── raw/                       # Original, immutable data
│   │   ├── cancer_patients/       # Cancer patients dataset
│   │   └── xray_images/           # X-ray images dataset
│   ├── processed/                 # Cleaned and preprocessed data
│   ├── features/                  # Feature engineering outputs
│   ├── interim/                   # Intermediate processing data
│   └── external/                  # External data sources
│
├── models/                        # Model implementations
│   ├── clustering/                # Clustering models
│   │   └── kmeans.py              # K-means implementation
│   ├── dimensionality_reduction/  # Dimensionality reduction models
│   │   └── pca.py                 # PCA implementation
│   ├── classification/            # Classification models
│   │   ├── random_forest.py       # Random Forest implementation
│   │   └── xgboost_model.py       # XGBoost implementation
│   ├── regression/                # Regression models
│   ├── time_series/               # Time series models
│   │   └── prophet.py             # Prophet implementation
│   └── ensemble/                  # Ensemble models
│
├── notebooks/                     # Jupyter notebooks for exploration and reporting
│   ├── exploration/               # Data exploration notebooks
│   ├── modeling/                  # Model development notebooks
│   └── evaluation/                # Model evaluation notebooks
│
├── src/                           # Source code
│   ├── data/                      # Data processing code
│   │   ├── make_dataset.py        # Scripts to download/generate data
│   │   └── preprocessing.py       # Data cleaning and preprocessing
│   ├── features/                  # Feature engineering code
│   │   └── build_features.py      # Feature extraction and transformation
│   ├── models/                    # Model training and prediction code
│   │   ├── train_model.py         # Model training scripts
│   │   └── predict_model.py       # Model prediction scripts
│   ├── visualization/             # Visualization code
│   │   └── visualize.py           # Visualization utilities
│   └── utils/                     # Utility functions
│       ├── evaluation.py          # Model evaluation utilities
│       └── logging.py             # Logging configuration
│
├── tests/                         # Testing code
│   ├── test_data.py               # Tests for data processing
│   └── test_models.py             # Tests for models
│
├── outputs/                       # Output files
│   ├── models/                    # Trained model files
│   │   └── serialized/            # Serialized model objects
│   ├── figures/                   # Generated plots and figures
│   └── results/                   # Analysis results
│
├── docs/                          # Documentation
│   ├── data_dictionary.md         # Description of data fields
│   ├── model_specs.md             # Model specifications
│   └── setup_guide.md             # Setup instructions
│
├── app/                           # Application code (if building a demo or service)
│   ├── api/                       # API endpoints
│   ├── frontend/                  # Frontend code if applicable
│   └── main.py                    # Main application entry point
│
├── scripts/                       # Utility scripts
│   ├── setup_environment.sh       # Environment setup script
│   └── download_datasets.py       # Dataset download scripts
│
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── environment.yml                # Conda environment file
├── requirements.txt               # Package dependencies
├── setup.py                       # Makes project pip installable
└── README.md                      # Project description and instructions