import numpy as np


def predict_future(model, last_sequence, scaler, steps=24):
    predictions = []

    current_seq = last_sequence.copy()

    for _ in range(steps):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape))[0][0]
        predictions.append(pred)

        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions
