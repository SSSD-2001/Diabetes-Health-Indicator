# Diabetes Health Indicator Prediction

A supervised classification project comparing **Logistic Regression** and **Random Forest** to predict diabetes presence based on health and lifestyle indicators.

## Project Overview

This project analyzes the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) to:
- Understand patterns and risk factors for diabetes
- Build and compare classification models
- Identify the most accurate algorithm for prediction
- Provide actionable insights from the analysis

## Dataset

The dataset includes various health and lifestyle factors such as:
- BMI, blood pressure, cholesterol levels
- Physical activity, fruit/vegetable consumption
- Smoking and alcohol use
- Medical history (heart disease, stroke, etc.)
- Target: Diabetes status (0=No, 1=Yes)

**Target Variable (Project Proposal)**: `Diabetes_binary`

**Source**: [Kaggle - Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Project Structure

```
Diabetes-Health-Indicator/
├── data/                          # Data directory
│   ├── raw/                      # Original datasets
│   └── processed/                # Cleaned/processed data
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Data cleaning & preprocessing
│   └── 03_model_training.ipynb   # Model training & comparison
├── src/                           # Python modules
│   ├── __init__.py
│   ├── preprocessing.py          # Data preprocessing functions
│   ├── models.py                 # Model training functions
├── results/                       # Model outputs
│   ├── models/                   # Trained model files
│   └── plots/                    # Visualization outputs
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Getting Started

### 1. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

- Visit [Kaggle Dataset Link](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- Download the CSV file
- Save it to `data/raw/` folder

### 3. Exploratory Data Analysis

Open and run the notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

Start exploring:
- Load and inspect the data
- Check for missing values
- Analyze feature distributions
- Examine correlations
- Identify class imbalance

### 4. Data Preprocessing

Follow the notebook `notebooks/02_preprocessing.ipynb`:
- Handle missing values
- Feature scaling/normalization
- Handle imbalanced classes (if needed)
- Train-test split

### 5. Model Training & Comparison

Use `notebooks/03_model_training.ipynb`:
- Train Logistic Regression
- Train Random Forest Classifier
- Compare performance metrics
- Cross-validation
- Hyperparameter tuning

### 6. Optional: Run a Web UI

This project also includes a simple Streamlit app for interactive predictions.

```bash
streamlit run streamlit_app.py
```

Then open:
- `http://localhost:8501`

## Key Metrics to Track

- **Accuracy**: Overall correctness
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Model's ability to distinguish classes
- **Confusion Matrix**: Breakdown of predictions

## Workflow

1. **Explore** → Understand data distribution and relationships
2. **Preprocess** → Clean, transform, and prepare data
3. **Train** → Build and compare models
4. **Compare** → Compare performance using multiple metrics
5. **Document** → Record findings and insights

## Models to Compare

### 1. Logistic Regression
- Linear baseline model
- Fast training & inference
- Good for interpretability

### 2. Random Forest
- Ensemble method
- Handles non-linear patterns
- Feature importance ranking

## Algorithm Selection Rationale (Guidelines Aligned)

This is a **supervised classification** task (predicting diabetes class). For this project, avoid ANN-based methods and use Scikit-Learn models.

Algorithms commonly available in Scikit-Learn that support both problem families (through classifier/regressor variants) include:
- Decision Tree (`DecisionTreeClassifier` / `DecisionTreeRegressor`)
- Random Forest (`RandomForestClassifier` / `RandomForestRegressor`)
- Support Vector Machine (`SVC` / `SVR`)
- K-Nearest Neighbors (`KNeighborsClassifier` / `KNeighborsRegressor`)
- Gradient Boosting (`GradientBoostingClassifier` / `GradientBoostingRegressor`)

Recommended pair for this dataset:
- **Logistic Regression** as an interpretable linear baseline
- **Random Forest Classifier** as a non-linear ensemble benchmark

Alternative strict pair (both with direct regressor counterparts):
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

Both choices satisfy the non-ANN constraint and are fully supported by Scikit-Learn.

## Performance Metrics (Classification)

Because medical-risk datasets are often imbalanced, do not rely only on accuracy.

Use these primary metrics:
- **Recall**: important to reduce false negatives (missed diabetes cases)
- **F1-Score**: balances precision and recall
- **ROC-AUC**: discrimination ability across thresholds
- **Precision**: confidence in positive predictions
- **Confusion Matrix**: error-type breakdown

Useful secondary metrics:
- **Balanced Accuracy**
- **PR-AUC** (especially if positive class is rare)

## Next Steps

- [ ] Download and explore dataset
- [ ] Run EDA notebook
- [ ] Preprocess data
- [ ] Train both models
- [ ] Compare results and document findings
- [ ] Create visualizations
- [ ] Write conclusions

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Notes

- Keep raw data in `data/raw/` untouched
- Save processed data in `data/processed/`
- Store trained models in `results/models/`
- Save plots and results in `results/plots/`

## License

[Add your license here]

## Author

[Your Name]
