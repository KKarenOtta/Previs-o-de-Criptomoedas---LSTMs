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

---

## 8. Descrição do desenvolvimento

8.1. src/data_loader.py → Coleta de dados
Busca dados históricos da API CoinGecko.
Entrada : coin="ethereum", days=120
Saída
    DataFrame:
    - price
    - volume
    - index = timestamp
Pipeline : fonte de dados do sistema

8.2. src/preprocessing.py → Preparação dos dados

    8.2.1 resample_data()
    Padroniza frequência temporal.
    df.resample("1h")
    - Evita buracos na série.
    
    8.2.2 add_technical_features()
    Cria variáveis derivadas:
    return_pct → variação percentual
    moving_average → tendência
    volatility → risco
    - Transforma preço bruto em sinais úteis.
   
    8.2.3 split_series_before_scaling()
    Divide em:
    train | validation | test
    - Importante: respeita ordem temporal.
   
    8.2.4 scale_train_val_test_features()
    Normaliza os dados:
    fit → apenas no treino
    transform → val/test
    - evita vazamento de dados (data leakage)
    
    8.2.5 create_sequences()
    Cria janelas temporais:
    [X1, X2, ..., Xt] → previsão de t+1
    Exemplo:
    window_size = 24
    → modelo vê últimas 24 horas
   
    8.2.6 build_temporal_sequences_without_leakage()
    Pipeline completo de preparação:
    split → scale → sequences → split final

8.3. src/model_lstm.py → Modelo LSTM
Rede neural recorrente (RNN).
Aprende padrões ao longo do tempo.
Entrada: (X_samples, window, features)
Saída: previsão do próximo valor
É o modelo principal do projeto

8.4. src/train.py → Treinamento
Treina modelos (LSTM principalmente).
Inclui
    EarlyStopping
    ModelCheckpoint
evita overfitting
salva melhor modelo automaticamente

8.5. src/models_mirofish/ → Modelo alternativo
    8.5.1 mirofish_model.py
    Define arquitetura PyTorch.
    8.5.2 train_mirofish.py
    Treina modelo.
    8.5.3 evaluate_mirofish.py
    Avalia modelo.

8.6. src/evaluate.py → Métricas
Calcula
    MSE
    RMSE
    MAE
    R²

Quantifica a qualidade da previsão

8.7. src/predict.py → Previsão futura
Responsável pelo: predict_future(...)
Gera valores futuros iterativamente.

8.8. src/utils.py → Utilidades
Principais funções
    set_global_seed()
    Reprodutibilidade
    validate_series_dataframe()
    Valida dados
    build_rmse_weighted_ensemble()
Combina modelos
    plot_model_comparison()
Gera gráficos

8.9. src/pipeline.py → Orquestrador
Executa: dados → preprocess → treino → avaliação → ensemble

run_full_pipeline()
Saída:
     metrics
     modelos
     previsões
     gráficos

8.10. reports/generate_data_report.py → Relatório
Gera:
    métricas
    gráficos
    análise
Saída: reports/relatorio_completo.md

8.11. Notebook (notebooks/)
Interface interativa:
    testar ideias
    visualizar dados
    debug

8.12. Fluxo completo resumido
    a. data_loader → coleta dados
    b. preprocessing → limpa e cria features
    c. sequences → cria janelas temporais
    d. LSTM → aprende padrões
    e. MiroFish → modelo alternativo
    f. evaluate → mede erro
    g. ensemble → combina modelos
    h. report → gera análise final
    
