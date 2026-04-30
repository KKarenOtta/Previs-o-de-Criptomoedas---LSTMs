# Previsão de Criptomoedas com LSTM e MiroFish

Projeto de Redes Neurais para previsão de preços de criptomoedas usando séries temporais, redes LSTM em TensorFlow/Keras e modelo MiroFish em PyTorch.

O objetivo é comparar abordagens neurais para previsão do preço do Ethereum a partir de dados históricos da API CoinGecko.

---

## 1. Stack principal

### Linguagem
- Python 3.12

### Machine Learning/ RN
- TensorFlow / Keras
- PyTorch
- Scikit-learn

### Dados
- CoinGecko API
- Pandas
- NumPy

### Visualização
- Matplotlib

### Ambiente interativo
- Jupyter Notebook
- IPython Kernel

---

## 2. Estrutura do projeto
    
    .
    ├── README.md
    ├── requirements.txt
    ├── generate_notebook.py
    ├── notebooks
    │   └── lstm_pipeline.ipynb
    ├── src
    │   ├── data_loader.py
    │   ├── evaluate.py
    │   ├── model_lstm.py
    │   ├── pipeline.py
    │   ├── predict.py
    │   ├── preprocessing.py
    │   ├── train.py
    │   ├── utils.py
    │   └── models_mirofish
    │       ├── evaluate_mirofish.py
    │       ├── mirofish_model.py
    │       └── train_mirofish.py
    └── outputs
        ├── models
        └── plots


## 3. O que cada parte faz
        src/data_loader.py
        Responsável por buscar dados históricos de criptomoedas na API CoinGecko.
        Função principal:
        get_historical_data(coin="ethereum", days=120)
        Retorna um DataFrame com:
        índice temporal
        preço em USD
        
        src/preprocessing.py
        Responsável pelo preparo da série temporal.
        Inclui:
        reamostragem por hora
        divisão temporal treino/validação/teste
        normalização com MinMaxScaler
        criação de janelas temporais para modelos LSTM
        Ponto importante:
        O scaler deve ser ajustado apenas no treino para evitar vazamento de dados.
        
        src/model_lstm.py
        Define o modelo LSTM em TensorFlow/Keras.
        O modelo aprende padrões temporais da série e retorna uma previsão de preço normalizado para o próximo passo.
        
        src/models_mirofish/
        Contém a implementação alternativa em PyTorch.
        Arquivos principais:
        mirofish_model.py
        train_mirofish.py
        evaluate_mirofish.py
        O MiroFish também usa LSTM internamente, mas com treinamento e persistência em PyTorch.
        
        src/train.py
        Treina a LSTM com:
        EarlyStopping
        ModelCheckpoint
        salvamento do melhor modelo em outputs/models/
        gráfico de perda em outputs/plots/
        
        src/evaluate.py
        Avalia o modelo LSTM com métricas:
        MSE
        RMSE
        MAE
        R²
        
        src/predict.py
        Faz previsão futura usando o último bloco temporal conhecido.
        
        src/utils.py
        Contém funções auxiliares reutilizáveis:
        criação de diretórios
        seed global
        validação de DataFrame
        comparação de métricas
        gráficos comparativos
        
        src/pipeline.py
        Pipeline principal do projeto.
        Integra:
        coleta de dados
        pré-processamento
        treino da LSTM
        treino do MiroFish
        avaliação
        comparação
        geração de gráficos
        persistência dos modelos

## 4. Como preparar o ambiente
        
        4.1. Copiar o projeto
        git clone https://github.com/KKarenOtta/Previs-o-de-Criptomoedas---LSTMs
       
        4.2. Criar ambiente virtual com Python 3.12
        python3.12 -m venv .venv
        source .venv/bin/activate
       
        4.3. Atualizar ferramentas base
        python -m pip install --upgrade pip setuptools wheel
       
        4.4. Instalar dependências
        python -m pip install -r requirements.txt

## 5. Como validar instalação

    Execute:
   
        python - <<'PY'
        import tensorflow as tf
        import torch
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        print("TensorFlow:", tf.__version__)
        print("Torch:", torch.__version__)
        print("Pandas:", pd.__version__)
        print("NumPy:", np.__version__)
        print("Scikit-learn:", sklearn.__version__)
        print("Matplotlib:", matplotlib.__version__)
        print("Ambiente OK")
        PY
   
Se tudo estiver correto, o terminal deve imprimir as versões sem erro.

##6. Como testar sintaxe do projeto

     python -m py_compile \
     generate_notebook.py \
     src/data_loader.py \
     src/evaluate.py \
     src/model_lstm.py \
     src/pipeline.py \
     src/predict.py \
     src/preprocessing.py \
     src/train.py \
     src/utils.py \
     src/models_mirofish/evaluate_mirofish.py \
     src/models_mirofish/mirofish_model.py \
     src/models_mirofish/train_mirofish.py
Se não aparecer erro, todos os arquivos Python estão sintaticamente válidos.

## 7. Como rodar o pipeline pelo terminal
Teste rápido:

    python - <<'PY'
    from src.pipeline import run_full_pipeline
    
    result = run_full_pipeline(
       coin="ethereum",
       days=120,
       window_size=24,
       epochs=5,
       batch_size=32,
    )
    
    print(result["metrics"])
    print("LSTM salvo em:", result["lstm_model_path"])
    print("MiroFish salvo em:", result["mirofish_model_path"])
    print("Gráfico salvo em:", result["comparison_plot"])
    PY
Saídas esperadas:
tabela de métricas
modelo LSTM salvo em outputs/models/lstm_model.keras
modelo MiroFish salvo em outputs/models/mirofish_model.pt
gráfico salvo em outputs/plots/comparison_lstm_mirofish.png

## 8. Como rodar no Jupyter Notebook
8.1. Registrar kernel

    python -m ipykernel install --user \
     --name previsao-cripto-lstm \
     --display-name "Previsao Cripto LSTM"

8.2. Abrir Jupyter

    jupyter notebook

8.3. Abrir notebook

    Abra:
        notebooks/lstm_pipeline.ipynb
    No menu do notebook, selecione:
        Kernel > Change Kernel > Previsao Cripto LSTM
    Depois rode:
        Kernel > Restart Kernel and Run All Cells

