import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 10         
GRU_UNITS_1 = 64
GRU_UNITS_2 = 32
DROPOUT_RATE = 0.2
PLOT_SAMPLES = 500     

OUTPUT_DIR = config.OUTPUT_DIR / "gru"


def build_model(window, features=1):
    model = keras.Sequential([
        layers.Input(shape=(window, features)),
        layers.GRU(GRU_UNITS_1, return_sequences=True),
        layers.GRU(GRU_UNITS_2),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def train(model, train_X, train_y, test_X, test_y):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        train_X, train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_X, test_y),
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def evaluate(model, test_X, test_y):
    pred_y = model.predict(test_X, batch_size=BATCH_SIZE, verbose=0).flatten()
    mae = mean_absolute_error(test_y, pred_y)
    rmse = float(np.sqrt(mean_squared_error(test_y, pred_y)))
    r2 = r2_score(test_y, pred_y)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_loss(history, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("GRU — Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gru_loss.png", dpi=150)
    plt.close()


def plot_predictions(model, test_X, test_y, out_dir, n=PLOT_SAMPLES):
    pred_y = model.predict(test_X[:n], batch_size=BATCH_SIZE, verbose=0).flatten()
    plt.figure(figsize=(12, 4))
    plt.plot(test_y[:n], label="Actual", alpha=0.7)
    plt.plot(pred_y, label="Predicted", alpha=0.7)
    plt.xlabel("Sample index")
    plt.ylabel("Normalised volume")
    plt.title(f"GRU — Predicted vs Actual (first {n} test samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gru_predictions.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    train_X = np.load(config.OUTPUT_DIR / "train_X.npy")
    train_y = np.load(config.OUTPUT_DIR / "train_y.npy")
    test_X  = np.load(config.OUTPUT_DIR / "test_X.npy")
    test_y  = np.load(config.OUTPUT_DIR / "test_y.npy")
    print(f"  train: X={train_X.shape}  y={train_y.shape}")
    print(f"  test:  X={test_X.shape}  y={test_y.shape}")

    window   = train_X.shape[1]
    features = train_X.shape[2]

    print("\nBuilding model ...")
    model = build_model(window, features)
    model.summary()

    print("\nTraining ...")
    history = train(model, train_X, train_y, test_X, test_y)
    epochs_trained = len(history.history["loss"])

    print("\nEvaluating ...")
    metrics = evaluate(model, test_X, test_y)
    metrics["epochs_trained"] = epochs_trained
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  R²   : {metrics['r2']:.6f}")
    print(f"  Epochs trained: {epochs_trained}")

    print("\nSaving outputs ...")
    model.save(OUTPUT_DIR / "gru_model.keras")

    with open(OUTPUT_DIR / "gru_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_loss(history, OUTPUT_DIR)
    plot_predictions(model, test_X, test_y, OUTPUT_DIR)

    print(f"\nDone. Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
