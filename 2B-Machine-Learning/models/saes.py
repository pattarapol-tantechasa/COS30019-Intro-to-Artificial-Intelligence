import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential

import config

# SAEs-specific hyperparameters
EPOCHS_PRETRAIN = 50       # epochs for each individual autoencoder
EPOCHS_FINETUNE = 100      # epochs for the final stacked model fine-tuning
BATCH_SIZE = 256
LEARNING_RATE = 0.001
PATIENCE = 10              # early stopping patience (fine-tuning phase)
# Layer sizes: [input, hidden1, hidden2, hidden3, output]
# Input = WINDOW (12), three hidden layers, output = 1
SAE_LAYERS = [config.WINDOW, 400, 400, 400, 1]
PLOT_SAMPLES = 500         # number of test samples shown in predictions plot

OUTPUT_DIR = config.OUTPUT_DIR / "saes"


# ---------------------------------------------------------------------------
# Model construction  (adapted from SAEs sample implementation code in model.py)
# ---------------------------------------------------------------------------

def _build_single_sae(inputs, hidden, output):
    """Build one shallow autoencoder used during greedy pre-training.

    Structure:  input -> hidden (sigmoid) -> dropout -> output (sigmoid)
    The 'hidden' layer name is fixed so we can extract its weights later
    and transplant them into the stacked model.
    """
    model = Sequential([
        Dense(hidden, input_dim=inputs, name='hidden'),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(output, activation='sigmoid'),
    ])
    return model


def build_saes(layer_sizes):
    """Build the list of models used for SAE training.

    Returns [sae1, sae2, sae3, saes] where:
      - sae1/sae2/sae3 are individual autoencoders used in pre-training
      - saes is the final stacked model that gets fine-tuned

    layer_sizes: [input_dim, h1, h2, h3, output_dim]
    """
    sae1 = _build_single_sae(layer_sizes[0], layer_sizes[1], layer_sizes[-1])
    sae2 = _build_single_sae(layer_sizes[1], layer_sizes[2], layer_sizes[-1])
    sae3 = _build_single_sae(layer_sizes[2], layer_sizes[3], layer_sizes[-1])

    # Stacked model — three hidden layers + dropout + output
    saes = Sequential([
        Dense(layer_sizes[1], input_dim=layer_sizes[0], name='hidden1'),
        Activation('sigmoid'),
        Dense(layer_sizes[2], name='hidden2'),
        Activation('sigmoid'),
        Dense(layer_sizes[3], name='hidden3'),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(layer_sizes[4], activation='sigmoid'),
    ])

    return [sae1, sae2, sae3, saes]


# ---------------------------------------------------------------------------
# Greedy layer-wise pre-training  (adapted from train.py sample code)
# ---------------------------------------------------------------------------

def pretrain(models, train_X, train_y):
    """Train each autoencoder greedily, then transfer weights to the stacked model.

    For sae_i (i > 0), the input is the hidden-layer output of sae_{i-1},
    not the raw data — that's the 'stacking' part.

    Returns the final fine-tunable stacked model (not yet fine-tuned).
    """
    temp = train_X.copy()

    for i in range(len(models) - 1):   # iterate over sae1, sae2, sae3
        # From sae2 onwards, pass data through the previous SAE's hidden layer
        if i > 0:
            prev = models[i - 1]
            # Extract hidden layer output by building a new model from scratch.
            # Keras 3 requires the model to have been called before accessing
            # .input, so we build a small encoder using the trained weights.
            prev_hidden = prev.get_layer('hidden')
            encoder = Sequential([
                Dense(prev_hidden.units, input_dim=temp.shape[1], name='hidden'),
                Activation('sigmoid'),
            ])
            encoder.get_layer('hidden').set_weights(prev_hidden.get_weights())
            temp = encoder.predict(temp, verbose=0)

        sae = models[i]
        sae.compile(loss='mse', optimizer=keras.optimizers.Adam(LEARNING_RATE))
        print(f"  Pre-training SAE {i + 1}/{len(models) - 1} "
              f"(input dim: {temp.shape[1]}) ...")
        sae.fit(
            temp, train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS_PRETRAIN,
            validation_split=0.05,
            verbose=0,          # quiet — progress shown at fine-tune stage
        )
        models[i] = sae

    # Transfer the learned hidden-layer weights into the stacked model
    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i + 1}').set_weights(weights)

    return saes


def finetune(saes, train_X, train_y, test_X, test_y):
    """Fine-tune the full stacked model end-to-end."""
    saes.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history = saes.fit(
        train_X, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_FINETUNE,
        validation_data=(test_X, test_y),
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation and plots  (same structure as lstm.py)
# ---------------------------------------------------------------------------

def evaluate(model, test_X, test_y):
    pred_y = model.predict(test_X, batch_size=BATCH_SIZE, verbose=0).flatten()
    mae  = mean_absolute_error(test_y, pred_y)
    rmse = float(np.sqrt(mean_squared_error(test_y, pred_y)))
    r2   = r2_score(test_y, pred_y)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_loss(history, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'],     label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('SAEs — Fine-tune Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'saes_loss.png', dpi=150)
    plt.close()


def plot_predictions(model, test_X, test_y, out_dir, n=PLOT_SAMPLES):
    pred_y = model.predict(test_X[:n], batch_size=BATCH_SIZE, verbose=0).flatten()
    plt.figure(figsize=(12, 4))
    plt.plot(test_y[:n], label='Actual',    alpha=0.7)
    plt.plot(pred_y,     label='Predicted', alpha=0.7)
    plt.xlabel('Sample index')
    plt.ylabel('Normalised volume')
    plt.title(f'SAEs — Predicted vs Actual (first {n} test samples)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'saes_predictions.png', dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading data ...')
    train_X = np.load(config.OUTPUT_DIR / 'train_X.npy')
    train_y = np.load(config.OUTPUT_DIR / 'train_y.npy')
    test_X  = np.load(config.OUTPUT_DIR / 'test_X.npy')
    test_y  = np.load(config.OUTPUT_DIR / 'test_y.npy')
    print(f'  train: X={train_X.shape}  y={train_y.shape}')
    print(f'  test:  X={test_X.shape}  y={test_y.shape}')

    # SAEs expect flat 2-D input (samples, timesteps) — not 3-D like LSTM/GRU.
    # The data pipeline adds a trailing features axis; squeeze it off here.
    train_X_2d = train_X.squeeze(-1)   # (samples, 12)
    test_X_2d  = test_X.squeeze(-1)    # (samples, 12)
    print(f'  reshaped for SAEs: train={train_X_2d.shape}  test={test_X_2d.shape}')

    print('\nBuilding models ...')
    models = build_saes(SAE_LAYERS)
    models[-1].summary()   # print the stacked model architecture

    print('\nPre-training (greedy layer-wise) ...')
    saes = pretrain(models, train_X_2d, train_y)

    print('\nFine-tuning stacked model ...')
    history = finetune(saes, train_X_2d, train_y, test_X_2d, test_y)
    epochs_trained = len(history.history['loss'])

    print('\nEvaluating ...')
    metrics = evaluate(saes, test_X_2d, test_y)
    metrics['epochs_finetuned'] = epochs_trained
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  R²   : {metrics['r2']:.6f}")
    print(f"  Fine-tune epochs: {epochs_trained}")

    print('\nSaving outputs ...')
    saes.save(OUTPUT_DIR / 'saes_model.keras')

    with open(OUTPUT_DIR / 'saes_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_loss(history, OUTPUT_DIR)
    plot_predictions(saes, test_X_2d, test_y, OUTPUT_DIR)

    print(f'\nDone. Files saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
