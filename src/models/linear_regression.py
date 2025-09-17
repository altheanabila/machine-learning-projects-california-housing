import numpy as np
from typing import Tuple, Optional




def closed_form_solution(X_b: np.ndarray, y: np.ndarray) -> np.ndarray:


    theta = np.linalg.pinv(X_b).dot(y)
    return theta




def gradient_descent(X_b: np.ndarray, y: np.ndarray,
        lr: float = 0.01, n_iters: int = 5000,
        return_history: bool = False) -> Tuple[np.ndarray, Optional[list]]:

    m, n = X_b.shape
    theta = np.zeros(n)
    history = []


    for it in range(n_iters):
        predictions = X_b.dot(theta)
        error = predictions - y
        gradients = (2.0 / m) * X_b.T.dot(error)
        theta -= lr * gradients


        if return_history and (it % max(1, n_iters // 100) == 0):
            loss = (error ** 2).mean()
            history.append((it, loss))


    if return_history:
        return theta, history
    else:
        return theta, None




def predict(X_b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X_b.dot(theta)