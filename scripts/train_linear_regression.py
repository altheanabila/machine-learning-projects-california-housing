from src.data_loader import load_california_housing
from src.preprocessing import train_test_scale_add_bias
from src.models.linear_regression import closed_form_solution, gradient_descent, predict
from src.evaluation import mse, rmse, r2, plot_pred_vs_actual, plot_residuals




def main():

    X, y, meta = load_california_housing(as_frame=True)
    print('Dataset shape:', X.shape)



    X_train_b, X_test_b, y_train, y_test, scaler = train_test_scale_add_bias(X, y)



    theta_closed = closed_form_solution(X_train_b, y_train)
    y_pred_closed = predict(X_test_b, theta_closed)
    print('Closed-form MSE:', mse(y_test, y_pred_closed))
    print('Closed-form RMSE:', rmse(y_test, y_pred_closed))
    print('Closed-form R2:', r2(y_test, y_pred_closed))


    plot_pred_vs_actual(y_test, y_pred_closed)
    plot_residuals(y_test, y_pred_closed)



    theta_gd, history = gradient_descent(X_train_b, y_train, lr=0.01, n_iters=5000, return_history=True)
    y_pred_gd = predict(X_test_b, theta_gd)
    print('GD MSE:', mse(y_test, y_pred_gd))
    print('GD RMSE:', rmse(y_test, y_pred_gd))
    print('GD R2:', r2(y_test, y_pred_gd))


    plot_pred_vs_actual(y_test, y_pred_gd)
    plot_residuals(y_test, y_pred_gd)



    import numpy as np
    mu = scaler.mean_
    sigma = scaler.scale_
    theta_s = theta_closed
    theta_original = np.empty_like(theta_s)
    theta_original[1:] = theta_s[1:] / sigma
    theta_original[0] = theta_s[0] - np.sum((theta_s[1:] * mu) / sigma)
    print('\nClosed-form coefficients (original scale):')
    print('Intercept:', theta_original[0])

    for fname, coef in zip(X.columns, theta_original[1:]):
        print(f" {fname}: {coef:.6f}")




if __name__ == '__main__':
    main()