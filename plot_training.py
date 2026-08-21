
# PART 6: TRAINING LOSS CURVE
# English -> Hindi NMT

import os
import json
import matplotlib.pyplot as plt

# CONFIGURATION

HISTORY_FILE = "artifacts/training_history.json"
OUTPUT_DIR = "artifacts"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "training_loss_curve.png"
)

# LOAD TRAINING HISTORY

def load_training_history():
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(
            f"Training history not found:\n"
            f"{HISTORY_FILE}\n\n"
            f"Make sure train.py has completed successfully."
        )
    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        history = json.load(file)
    print("=" * 70)
    print("TRAINING HISTORY LOADED")
    print("=" * 70)
    print(
        f"History file: {HISTORY_FILE}"
    )
    return history

# DISPLAY HISTORY

def display_history(history):
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    train_losses = history.get(
        "train_loss",
        []
    )
    val_losses = history.get(
        "val_loss",
        []
    )
    print(
        f"Number of training epochs: "
        f"{len(train_losses)}"
    )
    for i in range(len(train_losses)):
        train_loss = train_losses[i]
        if i < len(val_losses):
            val_loss = val_losses[i]
        else:
            val_loss = None
        if val_loss is not None:
            print(
                f"Epoch {i + 1}: "
                f"Train Loss = {train_loss:.4f}, "
                f"Validation Loss = {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {i + 1}: "
                f"Train Loss = {train_loss:.4f}"
            )

# CREATE LOSS CURVE

def create_loss_curve(history):
    train_losses = history.get(
        "train_loss",
        []
    )
    val_losses = history.get(
        "val_loss",
        []
    )
    if len(train_losses) == 0:
        raise ValueError(
            "No training loss values found."
        )
    epochs = range(
        1,
        len(train_losses) + 1
    )
    plt.figure(
        figsize=(10, 6)
    )
    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )
    if len(val_losses) > 0:
        plt.plot(
            epochs,
            val_losses,
            marker="o",
            label="Validation Loss"
        )
    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "Loss"
    )
    plt.title(
        "English-to-Hindi NMT Training and Validation Loss"
    )
    plt.legend()
    plt.grid(
        True
    )
    plt.tight_layout()
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    plt.savefig(
        OUTPUT_FILE,
        dpi=300
    )
    plt.show()
    print("\n" + "=" * 70)
    print("LOSS CURVE CREATED")
    print("=" * 70)
    print(
        f"Saved to:\n"
        f"{os.path.abspath(OUTPUT_FILE)}"
    )

# MAIN

def main():
    history = load_training_history()
    display_history(
        history
    )
    create_loss_curve(
        history
    )
    print("\n" + "=" * 70)
    print("PART 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)

# RUN

if __name__ == "__main__":
    main()