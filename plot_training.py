
# PART 3: TRAINING LOSS CURVE
# English -> Hindi NMT
import os
import json
import matplotlib.pyplot as plt
# CONFIGURATION
HISTORY_FILE = "artifacts/training_history.json"
ARTIFACTS_DIR = "artifacts"
OUTPUT_FILE = os.path.join(
    ARTIFACTS_DIR,
    "training_loss_curve.png"
)
# LOAD TRAINING HISTORY
def load_history():
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(
            f"Training history not found: {HISTORY_FILE}"
        )
    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        history = json.load(file)
    return history
# GENERATE LOSS CURVE
def plot_loss_curve(history):
    train_loss = history["train_loss"]
    validation_loss = history["validation_loss"]
    epochs = range(
        1,
        len(train_loss) + 1
    )
    plt.figure(
        figsize=(8, 5)
    )
    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Training Loss"
    )
    plt.plot(
        epochs,
        validation_loss,
        marker="o",
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "English → Hindi NMT Training Loss"
    )
    plt.legend()
    plt.grid(
        True
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE,
        dpi=300
    )
    plt.show()
    print(
        f"\nLoss curve saved at:"
        f"\n{OUTPUT_FILE}"
    )
# MAIN
def main():
    print("=" * 70)
    print("GENERATING TRAINING LOSS CURVE")
    print("=" * 70)
    history = load_history()
    print(
        f"\nEpochs: {len(history['train_loss'])}"
    )
    print(
        "\nTraining losses:"
    )
    for i, loss in enumerate(
        history["train_loss"],
        start=1
    ):
        print(
            f"Epoch {i}: {loss:.4f}"
        )
    print(
        "\nValidation losses:"
    )
    for i, loss in enumerate(
        history["validation_loss"],
        start=1
    ):
        print(
            f"Epoch {i}: {loss:.4f}"
        )
    plot_loss_curve(
        history
    )
    print("\n" + "=" * 70)
    print("LOSS CURVE GENERATED SUCCESSFULLY")
    print("=" * 70)
# RUN
if __name__ == "__main__":
    main()