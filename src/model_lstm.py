from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


def build_lstm_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
    )

    return model
