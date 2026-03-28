"""
Evaluation metrics and utilities for the Diabetes Health Indicator project.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report, roc_curve
)


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for ROC-AUC)
    
    Returns:
    --------
    dict
        Dictionary containing various evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def print_evaluation_report(metrics, model_name):
    """
    Print formatted evaluation report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from evaluate_model()
    model_name : str
        Name of the model
    """
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"{'='*50}\n")


def compare_models(results_dict):
    """
    Compare multiple models and display results.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'F1-Score':<12}")
    print(f"{'-'*70}")
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['f1']:<12.4f}")
    
    print(f"{'='*70}\n")
