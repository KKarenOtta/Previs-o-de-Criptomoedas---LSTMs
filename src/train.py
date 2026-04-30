from pathlib import Path

import matplotlib.pyplot as plt


def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.show()

    return model, history
