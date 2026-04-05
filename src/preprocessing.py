"""
Data preprocessing functions for the Diabetes Health Indicator project.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy to handle missing values ('drop', 'mean', 'median')
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def scale_features(X_train, X_test, method='standard'):
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Testing features
    method : str
        Scaling method ('standard' or 'minmax')
    
    Returns:
    --------
    tuple
        Scaled training and testing features
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features
    y : pd.Series or np.ndarray
        Target variable
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    stratify : bool
        If True, preserve class distribution in train/test split (recommended for classification)
    
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
