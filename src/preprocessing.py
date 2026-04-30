import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = [
    "price",
    "volume",
    "return_pct",
    "moving_average",
    "volatility",
]


def resample_data(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame vazio. Não é possível reamostrar.")

    df = df.resample(freq).mean()
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()

    return df


def add_technical_features(
    df: pd.DataFrame,
    moving_average_window: int = 24,
    volatility_window: int = 24,
) -> pd.DataFrame:
    if "price" not in df.columns:
        raise ValueError("Coluna obrigatória ausente: price")

    if "volume" not in df.columns:
        df = df.copy()
        df["volume"] = 0.0

    features = df.copy()

    features["return_pct"] = features["price"].pct_change()
    features["moving_average"] = features["price"].rolling(
        window=moving_average_window,
        min_periods=1,
    ).mean()
    features["volatility"] = features["return_pct"].rolling(
        window=volatility_window,
        min_periods=1,
    ).std()

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()
    features = features.dropna()

    return features[FEATURE_COLUMNS]


def split_series_before_scaling(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    total = len(df)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    if train_end == 0 or val_end <= train_end or val_end >= total:
        raise ValueError("Dados insuficientes para split temporal antes da normalização.")

    return (
        df.iloc[:train_end],
        df.iloc[train_end:val_end],
        df.iloc[val_end:],
    )


def scale_train_val_test_features(train_df, val_df, test_df):
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = feature_scaler.fit_transform(train_df[FEATURE_COLUMNS])
    X_val_scaled = feature_scaler.transform(val_df[FEATURE_COLUMNS])
    X_test_scaled = feature_scaler.transform(test_df[FEATURE_COLUMNS])

    y_train_scaled = target_scaler.fit_transform(train_df[["price"]])
    y_val_scaled = target_scaler.transform(val_df[["price"]])
    y_test_scaled = target_scaler.transform(test_df[["price"]])

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    )


def create_sequences(features: np.ndarray, target: np.ndarray, window_size: int = 24):
    if len(features) != len(target):
        raise ValueError("Features e target precisam ter o mesmo tamanho.")

    if len(features) <= window_size:
        raise ValueError(
            f"Dados insuficientes para window_size={window_size}. "
            f"Recebido: {len(features)} registros."
        )

    X = []
    y = []

    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def build_temporal_sequences_without_leakage(
    train_df,
    val_df,
    test_df,
    window_size: int = 24,
):
    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        feature_scaler,
        target_scaler,
    ) = scale_train_val_test_features(train_df, val_df, test_df)

    X_all_scaled = np.concatenate([X_train_scaled, X_val_scaled, X_test_scaled], axis=0)
    y_all_scaled = np.concatenate([y_train_scaled, y_val_scaled, y_test_scaled], axis=0)

    X_all, y_all = create_sequences(
        X_all_scaled,
        y_all_scaled,
        window_size=window_size,
    )

    total = len(X_all)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    return {
        "X_train": X_all[:train_end],
        "y_train": y_all[:train_end],
        "X_val": X_all[train_end:val_end],
        "y_val": y_all[train_end:val_end],
        "X_test": X_all[val_end:],
        "y_test": y_all[val_end:],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }
