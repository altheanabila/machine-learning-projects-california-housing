# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_squared_error(y_true, y_pred)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred)

def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    resid = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=40, alpha=0.7)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.title('Residuals Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
