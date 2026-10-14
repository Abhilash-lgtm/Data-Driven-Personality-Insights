# Data-Driven Personality Insights (DDPI)

A machine learning and data analysis system designed to uncover behavioral and personality patterns, evaluate psychological indicators, and predict life satisfaction and happiness metrics from survey data.

---

## 🚀 Overview

DDPI combines statistical analysis, unsupervised clustering, and predictive regression/classification models with an interactive web dashboard:

- **End-to-End Data Pipeline**: Missing value imputation with `KNNImputer`, feature engineering, standardization, and outlier filtering.
- **Dimensionality Reduction & Clustering**: PCA and K-Means clustering for customer/personality segmentation.
- **Predictive Machine Learning**:
  - Regression models (Linear Regression, KNN Regressor, Random Forest, XGBoost) predicting continuous traits and satisfaction levels.
  - Classification models (Random Forest Classifier, XGBoost Classifier) predicting happiness categories with ROC/AUC evaluation.
- **Interactive Web Interface**: Complete dashboard (`index.html`, `style.css`, `app.js`) backed by Python's built-in HTTP server (`app.py`) for live prediction and visualizations.

---

## 📁 Repository Structure

```text
├── DDPI.py                     # Core data pipeline, model training, and evaluation script
├── app.py                      # Lightweight server for serving dashboard and live API endpoints
├── index.html                  # Interactive frontend web dashboard
├── style.css                   # Custom styles for the web UI
├── app.js                      # Client-side dynamic charts and dashboard logic
├── responses_data.csv          # Survey dataset
├── results.json                # Persisted metrics and model evaluation outputs
├── requirements.txt            # Python dependencies
├── eda_correlation_heatmap.png # Correlation matrix visualization
├── eda_target_distribution.png # Distribution of target variables
├── model_comparison_mae.png    # Regression model comparison (MAE)
└── roc_curve_happiness.png     # ROC curve for happiness classification
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Abhilash-lgtm/Data-Driven-Personality-Insights.git
cd Data-Driven-Personality-Insights
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the ML Pipeline
To clean the dataset, train models, and generate analytical charts:
```bash
python DDPI.py
```

### 4. Launch the Web Application
Start the interactive local dashboard:
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## 📊 Outputs & Visualizations
- **Feature Correlation Matrix**: `eda_correlation_heatmap.png`
- **Target Distribution Plots**: `eda_target_distribution.png`
- **Model Performance Benchmark**: `model_comparison_mae.png`
- **ROC-AUC Performance**: `roc_curve_happiness.png`
- **Evaluation Metrics**: Exported automatically to `results.json`
