from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, PLOTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def validate_series_dataframe(df: pd.DataFrame, price_column: str = "price") -> None:
    if df.empty:
        raise ValueError("O DataFrame está vazio.")

    if price_column not in df.columns:
        raise ValueError(f"Coluna obrigatória ausente: {price_column}")

    if df[price_column].isna().all():
        raise ValueError(f"A coluna {price_column} contém apenas valores nulos.")

    if len(df) < 80:
        raise ValueError("Série histórica insuficiente. Use mais dias de dados.")


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics).T.sort_values(by="RMSE", ascending=True)


def build_rmse_weighted_ensemble(predictions: dict, metrics: dict):
    weights = {}

    for model_name, model_metrics in metrics.items():
        rmse = float(model_metrics["RMSE"])
        weights[model_name] = 1.0 / max(rmse, 1e-8)

    total_weight = sum(weights.values())
    weights = {name: value / total_weight for name, value in weights.items()}

    ensemble = None

    for model_name, pred in predictions.items():
        pred_array = np.asarray(pred).reshape(-1, 1)

        if ensemble is None:
            ensemble = weights[model_name] * pred_array
        else:
            ensemble += weights[model_name] * pred_array

    return ensemble, weights


def plot_model_comparison(y_true, predictions: dict, title: str, output_name: str):
    ensure_output_dirs()

    plt.figure(figsize=(12, 6))
    plt.plot(np.asarray(y_true).flatten(), label="Real")

    for label, values in predictions.items():
        plt.plot(np.asarray(values).flatten(), label=label)

    plt.title(title)
    plt.xlabel("Amostras de teste")
    plt.ylabel("Preço em USD")
    plt.legend()
    plt.tight_layout()

    output_path = PLOTS_DIR / output_name
    plt.savefig(output_path)
    plt.show()

    return output_path
