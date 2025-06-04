# Datasets Used in OncoPredictAI

## Primary Dataset: Global Cancer Patients Dataset (2015-2024)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024?resource=download)
- **Contents:** Patient demographics, cancer types, treatment outcomes, risk factors, and temporal trends.
- **Usage:** Core dataset for clustering, PCA, and predictive modeling.
- **Location:** Place CSV files in `data/` or `data/raw/cancer_patients/`.

## Secondary Dataset: Chest X-ray Pneumonia Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Contents:** Chest X-ray images labeled as normal or pneumonia.
- **Usage:** For future computer vision models (CNN, transfer learning).
- **Location:** Place images in `data/raw/xray_images/`.

## Download Instructions
- Use `scripts/download_dataset.py` to fetch datasets automatically (requires Kaggle API credentials).
- See `USER_GUIDE.md` for setup and troubleshooting.
