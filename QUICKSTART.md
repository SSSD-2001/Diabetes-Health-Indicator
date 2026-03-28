# Quick Start Guide

## Step 1: Download the Dataset

1. Go to [Kaggle: Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
2. Download the CSV file (might require Kaggle account)
3. Create a `data/raw/` folder if it doesn't exist
4. Save the CSV file as `diabetes_health_indicators.csv` in `data/raw/`

## Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Run the Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Open 01_eda.ipynb first
# Then run 02_model_training.ipynb
```

## Step 4: Understand Your Results

After running the notebooks, you'll have:
- **EDA insights**: Data distributions, missing values, correlations
- **Model comparison**: Performance metrics for both algorithms
- **Feature importance**: Which factors predict diabetes best
- **Visualizations**: Confusion matrices, ROC curves, performance charts

## Project Workflow

```
1. Exploratory Data Analysis (01_eda.ipynb)
   ↓
2. Data Understanding & Preprocessing
   ↓
3. Model Training & Comparison (02_model_training.ipynb)
   ↓
4. Results & Insights
   ↓
5. Documentation & Conclusions
```

## Expected Outcomes

- Train Logistic Regression on diabetes prediction
- Train Random Forest Classifier on diabetes prediction
- Compare accuracy, precision, recall, F1-score, and ROC-AUC
- Identify the best performing model
- Understand feature importance and health risk factors

## Key Files

- `notebooks/01_eda.ipynb` - Exploratory Data Analysis
- `notebooks/02_model_training.ipynb` - Model Training & Comparison
- `src/preprocessing.py` - Data preprocessing utilities
- `src/models.py` - Model training utilities
- `src/evaluation.py` - Evaluation metrics utilities
- `requirements.txt` - Python package dependencies

## Tips

1. **Start with EDA**: Understand your data before building models
2. **Check class balance**: Handle imbalanced data if needed
3. **Use cross-validation**: Ensures your model generalizes well
4. **Compare multiple metrics**: Don't just look at accuracy
5. **Save results**: Store trained models and visualizations

## Troubleshooting

**Missing dataset**: Make sure to download from Kaggle and save to `data/raw/`

**Import errors**: Reinstall dependencies with `pip install -r requirements.txt`

**Jupyter not starting**: Make sure you're in the virtual environment

## Next Steps After Running Notebooks

1. **Experiment with hyperparameters**: Tune the models for better performance
2. **Try more algorithms**: Add SVM, Gradient Boosting, Neural Networks
3. **Feature engineering**: Create new features from existing ones
4. **Ensemble methods**: Combine models for better predictions
5. **Deploy**: Build an API or web app for predictions
