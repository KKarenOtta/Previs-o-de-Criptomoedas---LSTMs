import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    y_test_inv = scaler.inverse_transform(y_test)
    pred_inv = scaler.inverse_transform(predictions)

    mse = mean_squared_error(y_test_inv, pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, pred_inv)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    }, y_test_inv, pred_inv
