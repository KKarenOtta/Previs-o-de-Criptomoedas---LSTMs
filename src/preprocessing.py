import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def resample_data(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    df = df.resample(freq).mean()
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()
    return df


def normalize_data(df: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler


def create_sequences(data: np.ndarray, window_size: int = 24):
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def train_test_split_time_series(X, y, train_ratio=0.7, val_ratio=0.15):
    total = len(X)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )
