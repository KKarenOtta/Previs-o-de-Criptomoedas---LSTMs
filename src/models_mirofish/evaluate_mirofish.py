import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_mirofish_model(model, X_test, y_test, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()

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
