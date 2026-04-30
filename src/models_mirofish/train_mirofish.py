from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def train_mirofish_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    lr=0.001,
    patience=8,
    model_name="mirofish_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "outputs" / "models"
    plots_dir = project_root / "outputs" / "plots"

    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_name

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    history = {"loss": [], "val_loss": []}

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        train_loss_value = float(loss.item())
        val_loss_value = float(val_loss.item())

        history["loss"].append(train_loss_value)
        history["val_loss"].append(val_loss_value)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss_value:.6f} | "
            f"Val Loss: {val_loss_value:.6f}"
        )

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            patience_counter = 0
            best_state = model.state_dict()
            torch.save(best_state, model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping aplicado na época {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training Loss - MiroFish")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss_mirofish.png")
    plt.show()

    return model, history, model_path
