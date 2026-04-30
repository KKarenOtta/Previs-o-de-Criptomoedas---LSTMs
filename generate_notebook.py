import json
import uuid
from pathlib import Path


def to_source(text):
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines]


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(code),
    }


cells = [
    markdown_cell("""
# Previsão de Preços de Criptomoedas com LSTM e MiroFish

Pipeline completo para previsão de preços do Ethereum usando dados históricos da API CoinGecko.
"""),

    code_cell("""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import get_historical_data
from src.preprocessing import (
    resample_data,
    normalize_data,
    create_sequences,
    train_test_split_time_series,
)
from src.model_lstm import build_lstm_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_future

from src.models_mirofish.mirofish_model import MiroFishModel
from src.models_mirofish.train_mirofish import train_mirofish_model
from src.models_mirofish.evaluate_mirofish import evaluate_mirofish_model

import matplotlib.pyplot as plt
"""),

    markdown_cell("""
## 1. Coleta dos dados
"""),

    code_cell("""
coin = "ethereum"
days = 120
window_size = 24

df = get_historical_data(coin=coin, days=days)
df.head()
"""),

    markdown_cell("""
## 2. Visualização inicial da série
"""),

    code_cell("""
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["price"], label="Preço real")
plt.title("Histórico de preços do Ethereum")
plt.xlabel("Data")
plt.ylabel("Preço em USD")
plt.legend()
plt.show()
"""),

    markdown_cell("""
## 3. Pré-processamento
"""),

    code_cell("""
df_resampled = resample_data(df, freq="1h")

scaled, scaler = normalize_data(df_resampled)

X, y = create_sequences(scaled, window_size=window_size)

X_train, y_train, X_val, y_val, X_test, y_test = train_test_split_time_series(
    X,
    y,
    train_ratio=0.7,
    val_ratio=0.15,
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
"""),

    markdown_cell("""
## 4. Modelo LSTM
"""),

    code_cell("""
lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
lstm_model.summary()
"""),

    markdown_cell("""
## 5. Treinamento da LSTM
"""),

    code_cell("""
lstm_model, history = train_model(
    lstm_model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=32,
)
"""),

    markdown_cell("""
## 6. Avaliação da LSTM
"""),

    code_cell("""
metrics_lstm, y_true_lstm, y_pred_lstm = evaluate_model(
    lstm_model,
    X_test,
    y_test,
    scaler,
)

metrics_lstm
"""),

    markdown_cell("""
## 7. Modelo MiroFish em PyTorch
"""),

    code_cell("""
mirofish_model = MiroFishModel(input_size=1)

mirofish_model = train_mirofish_model(
    mirofish_model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    lr=0.001,
)
"""),

    markdown_cell("""
## 8. Avaliação do MiroFish
"""),

    code_cell("""
metrics_mirofish, y_true_mirofish, y_pred_mirofish = evaluate_mirofish_model(
    mirofish_model,
    X_test,
    y_test,
    scaler,
)

metrics_mirofish
"""),

    markdown_cell("""
## 9. Comparação de métricas
"""),

    code_cell("""
print("LSTM")
print(metrics_lstm)

print("\\nMiroFish")
print(metrics_mirofish)
"""),

    markdown_cell("""
## 10. Gráfico comparativo
"""),

    code_cell("""
plt.figure(figsize=(12, 6))
plt.plot(y_true_lstm, label="Real")
plt.plot(y_pred_lstm, label="LSTM")
plt.plot(y_pred_mirofish, label="MiroFish")
plt.title("Comparação: Real vs LSTM vs MiroFish")
plt.xlabel("Amostras de teste")
plt.ylabel("Preço em USD")
plt.legend()
plt.show()
"""),

    markdown_cell("""
## 11. Predição futura com LSTM
"""),

    code_cell("""
future_lstm = predict_future(
    lstm_model,
    X_test[-1],
    scaler,
    steps=24,
)

plt.figure(figsize=(10, 5))
plt.plot(future_lstm, label="Previsão futura LSTM")
plt.title("Previsão futura dos próximos 24 passos")
plt.xlabel("Passo futuro")
plt.ylabel("Preço em USD")
plt.legend()
plt.show()
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path("notebooks/lstm_pipeline.ipynb")

with output_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Notebook regenerado corretamente: {output_path}")
