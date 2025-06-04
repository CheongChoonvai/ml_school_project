from setuptools import find_packages, setup

setup(
    name="oncopredict_ai",
    version="0.1.0",
    description="Machine Learning Framework for Global Cancer Data Analysis and Prediction",
    author="OncoPredictAI Team",
    packages=find_packages(include=["src", "src.*", "models", "models.*"]),
    install_requires=[
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "kagglehub>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.1.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
        "doc": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.9.0",
        ],
        "advanced_ml": [
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "prophet>=1.0.0",
        ],
    },
    python_requires=">=3.10",
)
