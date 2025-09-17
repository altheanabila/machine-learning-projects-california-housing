from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd




def train_test_scale_add_bias(X: pd.DataFrame, y: pd.Series,
test_size: float = 0.2, random_state: int = 42):

    X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=test_size, random_state=random_state
    )


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


# add bias column (intercept term)
    X_train_b = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_b = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]


    return X_train_b, X_test_b, y_train.to_numpy(), y_test.to_numpy(), scaler