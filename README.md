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
