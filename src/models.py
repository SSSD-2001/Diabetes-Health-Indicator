"""
Model training and utilities for the Diabetes Health Indicator project.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    **kwargs : dict
        Additional parameters for LogisticRegression
    
    Returns:
    --------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train Random Forest Classifier model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    **kwargs : dict
        Additional parameters for RandomForestClassifier
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model


def get_cross_validation_score(model, X, y, cv=5):
    """
    Get cross-validation score for a model.
    
    Parameters:
    -----------
    model : sklearn model
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    np.ndarray
        Cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores
