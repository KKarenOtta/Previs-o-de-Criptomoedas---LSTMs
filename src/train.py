import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("outputs/plots/training_loss.png")

    return model, history
