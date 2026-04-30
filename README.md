# Previsão de Criptomoedas com LSTM, MiroFish e Ensemble

Projeto de Machine Learning para previsão de preços de criptomoedas utilizando:

* LSTM (TensorFlow/Keras)
* MiroFish (PyTorch)
* Ensemble (combinação dos modelos)

---

## 1. Stack principal

### Linguagem
* Python 3.12

### Machine Learning/ Redes Neurais
* TensorFlow / Keras
* PyTorch
* Scikit-learn

### Dados
* CoinGecko API
* Pandas / NumPy

### Visualização
* Matplotlib

### Ambiente
* Jupyter Notebook

---

## 2. Arquitetura do sistema

Pipeline completo:

1. Coleta de dados
2. Feature engineering
3. Normalização
4. Criação de janelas temporais
5. Treinamento LSTM
6. Treinamento MiroFish
7. Ensemble
8. Avaliação
9. Geração de relatório

---

## 3. Features utilizadas

* Preço
* Retorno percentual
* Média móvel
* Volatilidade
* Volume (se disponível)

---

## 4. Como rodar o projeto

### 4.1 Setup

    ```bash
    git clone https://github.com/KKarenOtta/Previs-o-de-Criptomoedas---LSTMs
    cd Previs-o-de-Criptomoedas---LSTMs
    
    python3.12 -m venv .venv
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

---

### 4.2 Testar ambiente

    ```bash
    python - <<'PY'
    import tensorflow, torch, pandas, numpy
    print("Ambiente OK")
    PY
    ```

---

### 4.3 Rodar pipeline

    ```bash
    python - <<'PY'
    from src.pipeline import run_full_pipeline
    
    result = run_full_pipeline()
    print(result["metrics"])
    PY
    ```

---

### 4.4 Gerar relatório completo

    ```bash
    python reports/generate_data_report.py
    open reports/relatorio_completo.md
    ```

---

## 5. Métricas utilizadas

* MAE
* RMSE
* MAPE

---

## 6. Modelos

### LSTM

* Captura dependência temporal
* Suaviza série

### MiroFish

* Modelo experimental
* Menor capacidade temporal

### Ensemble

* Combinação dos modelos
* Melhor desempenho geral

---

## 7. Análise

* LSTM → melhor modelo individual
* MiroFish → underfitting + instabilidade
* Ensemble → melhor resultado geral
* Não usa dados externos (macro, notícias)
* Não é modelo financeiro real
* Não deve ser usado para investimento


