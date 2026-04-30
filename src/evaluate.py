import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_2d(array):
    return np.asarray(array).reshape(-1, 1)


def calculate_regression_metrics(y_true, y_pred):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }


def evaluate_model(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test)

    y_test_inv = target_scaler.inverse_transform(_as_2d(y_test))
    pred_inv = target_scaler.inverse_transform(_as_2d(predictions))

    metrics = calculate_regression_metrics(y_test_inv, pred_inv)

    return metrics, y_test_inv, pred_inv
