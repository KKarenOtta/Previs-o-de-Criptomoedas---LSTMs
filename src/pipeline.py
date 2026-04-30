from src.data_loader import get_historical_data
from src.evaluate import calculate_regression_metrics, evaluate_model
from src.model_lstm import build_lstm_model
from src.models_mirofish.evaluate_mirofish import evaluate_mirofish_model
from src.models_mirofish.mirofish_model import MiroFishModel
from src.models_mirofish.train_mirofish import train_mirofish_model
from src.preprocessing import (
    add_technical_features,
    build_temporal_sequences_without_leakage,
    resample_data,
    split_series_before_scaling,
)
from src.train import train_model
from src.utils import (
    build_rmse_weighted_ensemble,
    metrics_to_dataframe,
    plot_model_comparison,
    set_global_seed,
    validate_series_dataframe,
)


def prepare_time_series_data(
    coin="ethereum",
    days=120,
    window_size=24,
    freq="1h",
):
    df = get_historical_data(coin=coin, days=days)
    validate_series_dataframe(df)

    df_resampled = resample_data(df, freq=freq)
    validate_series_dataframe(df_resampled)

    df_features = add_technical_features(
        df_resampled,
        moving_average_window=24,
        volatility_window=24,
    )
    validate_series_dataframe(df_features)

    train_df, val_df, test_df = split_series_before_scaling(df_features)

    sequence_data = build_temporal_sequences_without_leakage(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        window_size=window_size,
    )

    return {
        "df": df,
        "df_resampled": df_resampled,
        "df_features": df_features,
        **sequence_data,
    }


def run_full_pipeline(
    coin="ethereum",
    days=120,
    window_size=24,
    epochs=50,
    batch_size=32,
):
    set_global_seed(42)

    data = prepare_time_series_data(
        coin=coin,
        days=days,
        window_size=window_size,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    target_scaler = data["target_scaler"]

    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    lstm_model, lstm_history, lstm_model_path = train_model(
        lstm_model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=8,
        model_name="lstm_model.keras",
    )

    metrics_lstm, y_true, y_pred_lstm = evaluate_model(
        lstm_model,
        X_test,
        y_test,
        target_scaler,
    )

    mirofish_model = MiroFishModel(input_size=X_train.shape[2])

    mirofish_model, mirofish_history, mirofish_model_path = train_mirofish_model(
        mirofish_model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        lr=0.005,
        patience=8,
        model_name="mirofish_model.pt",
    )

    metrics_mirofish, _, y_pred_mirofish = evaluate_mirofish_model(
        mirofish_model,
        X_test,
        y_test,
        target_scaler,
    )

    base_metrics = {
        "LSTM": metrics_lstm,
        "MiroFish": metrics_mirofish,
    }

    y_pred_ensemble, ensemble_weights = build_rmse_weighted_ensemble(
        predictions={
            "LSTM": y_pred_lstm,
            "MiroFish": y_pred_mirofish,
        },
        metrics=base_metrics,
    )

    metrics_ensemble = calculate_regression_metrics(
        y_true,
        y_pred_ensemble,
    )

    all_metrics = {
        **base_metrics,
        "Ensemble": metrics_ensemble,
    }

    metrics = metrics_to_dataframe(all_metrics)

    comparison_plot = plot_model_comparison(
        y_true,
        {
            "LSTM": y_pred_lstm,
            "MiroFish": y_pred_mirofish,
            "Ensemble": y_pred_ensemble,
        },
        title="Comparação: Real vs LSTM vs MiroFish vs Ensemble",
        output_name="comparison_lstm_mirofish_ensemble.png",
    )

    return {
        **data,
        "metrics": metrics,
        "raw_metrics": all_metrics,
        "ensemble_weights": ensemble_weights,
        "lstm_model": lstm_model,
        "lstm_history": lstm_history,
        "lstm_model_path": lstm_model_path,
        "mirofish_model": mirofish_model,
        "mirofish_history": mirofish_history,
        "mirofish_model_path": mirofish_model_path,
        "y_true": y_true,
        "y_pred_lstm": y_pred_lstm,
        "y_pred_mirofish": y_pred_mirofish,
        "y_pred_ensemble": y_pred_ensemble,
        "comparison_plot": comparison_plot,
    }
