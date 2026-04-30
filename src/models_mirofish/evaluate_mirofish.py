import numpy as np
import torch

from src.evaluate import calculate_regression_metrics


def _as_2d(array):
    return np.asarray(array).reshape(-1, 1)


def evaluate_mirofish_model(model, X_test, y_test, target_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    y_test_inv = target_scaler.inverse_transform(_as_2d(y_test))
    pred_inv = target_scaler.inverse_transform(_as_2d(predictions))

    metrics = calculate_regression_metrics(y_test_inv, pred_inv)

    return metrics, y_test_inv, pred_inv
