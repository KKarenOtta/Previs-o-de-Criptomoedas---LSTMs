from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_full_pipeline


# =========================
# EXECUTA PIPELINE
# =========================
result = run_full_pipeline(
    coin="ethereum",
    days=120,
    window_size=24,
    epochs=5,
    batch_size=32,
)

REPORT_DIR = PROJECT_ROOT / "reports"
IMG_DIR = REPORT_DIR / "images"
REPORT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)


# =========================
# DADOS
# =========================
df = result["df"]
df_features = result["df_features"]

y_true = result["y_true"]
y_lstm = result["y_pred_lstm"]
y_miro = result["y_pred_mirofish"]
y_ensemble = result["y_pred_ensemble"]

metrics = result["metrics"]


# =========================
# MÉTRICAS DETALHADAS
# =========================
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


metrics_lstm = compute_metrics(y_true, y_lstm)
metrics_miro = compute_metrics(y_true, y_miro)
metrics_ens = compute_metrics(y_true, y_ensemble)

metrics_df = pd.DataFrame(
    [metrics_lstm, metrics_miro, metrics_ens],
    index=["LSTM", "MiroFish", "Ensemble"],
)


# =========================
# GRÁFICOS
# =========================

# 1. comparação geral
plt.figure(figsize=(14,6))
plt.plot(y_true, label="Real")
plt.plot(y_lstm, label="LSTM")
plt.plot(y_miro, label="MiroFish")
plt.plot(y_ensemble, label="Ensemble")
plt.legend()
plt.title("Comparação de Modelos")
plt.savefig(IMG_DIR / "comparacao_modelos.png")
plt.close()


# 2. erro absoluto
plt.figure(figsize=(14,6))
plt.plot(np.abs(y_true - y_lstm), label="Erro LSTM")
plt.plot(np.abs(y_true - y_miro), label="Erro MiroFish")
plt.plot(np.abs(y_true - y_ensemble), label="Erro Ensemble")
plt.legend()
plt.title("Erro Absoluto por Modelo")
plt.savefig(IMG_DIR / "erro_modelos.png")
plt.close()


# 3. distribuição de erro
plt.figure(figsize=(10,5))
plt.hist(y_true - y_ensemble, bins=50)
plt.title("Distribuição do erro (Ensemble)")
plt.savefig(IMG_DIR / "distribuicao_erro.png")
plt.close()


# =========================
# RELATÓRIO
# =========================

report_path = REPORT_DIR / "relatorio_completo.md"

with open(report_path, "w") as f:

    f.write("# Relatório Completo de Modelagem - Criptomoedas\n\n")

    f.write("## 1. Visão Geral\n")
    f.write("- Modelo LSTM (Deep Learning)\n")
    f.write("- Modelo MiroFish (custom NN)\n")
    f.write("- Ensemble ponderado\n\n")

    f.write("## 2. Dados\n")
    f.write(f"- Total de registros: {len(df)}\n")
    f.write(f"- Features utilizadas: {list(df_features.columns)}\n\n")

    f.write("## 3. Métricas\n\n")
    f.write(metrics_df.to_markdown())
    f.write("\n\n")

    f.write("## 4. Análise dos Modelos\n\n")

    f.write("### LSTM\n")
    f.write("- Boa capacidade de suavização\n")
    f.write("- Pode perder picos (lag)\n\n")

    f.write("### MiroFish\n")
    f.write("- Underfitting evidente\n")
    f.write("- Baixa sensibilidade ao movimento\n\n")

    f.write("### Ensemble\n")
    f.write("- Melhor equilíbrio entre viés e variância\n")
    f.write("- Redução de erro médio\n\n")

    f.write("## 5. Gráficos\n\n")

    f.write("### Comparação\n")
    f.write("![](images/comparacao_modelos.png)\n\n")

    f.write("### Erro\n")
    f.write("![](images/erro_modelos.png)\n\n")

    f.write("### Distribuição de erro\n")
    f.write("![](images/distribuicao_erro.png)\n\n")

    f.write("## 6. Diagnóstico Técnico\n\n")

    f.write("- LSTM: estável, mas suaviza demais\n")
    f.write("- MiroFish: não está aprendendo dinâmica temporal\n")
    f.write("- Ensemble: melhor resultado atual\n\n")

    f.write("## 7. Recomendações\n\n")

    f.write("- Aumentar epochs (>=30)\n")
    f.write("- Adicionar mais features (RSI, MACD)\n")
    f.write("- Ajustar arquitetura MiroFish\n")
    f.write("- Testar Transformer (próximo passo)\n")


print(f"Relatório completo gerado em: {report_path}")
