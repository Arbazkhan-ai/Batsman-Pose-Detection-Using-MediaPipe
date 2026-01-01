# train_cricket_shot_models.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "cricket_shots_data.csv"
SEQ_LEN = 30            # number of frames per sequence (tweakable)
STRIDE = 5              # sliding window stride (overlap)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 32
EPOCHS = 60
RANDOM_STATE = 42
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# HELPERS
# -------------------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # First column is class label
    labels = df.iloc[:, 0].astype(str).values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return df, labels, features

def create_sequences(labels, features, seq_len=SEQ_LEN, stride=STRIDE):
    """
    Create sequences only from contiguous frames with the same label.
    Returns X (n_seq, seq_len, n_features) and y (n_seq,)
    """
    X_seqs = []
    y_seqs = []
    n_rows = features.shape[0]
    i = 0
    while i <= n_rows - seq_len:
        # check if the window has same label across its frames
        window_labels = labels[i:i+seq_len]
        if np.all(window_labels == window_labels[0]):
            X_seqs.append(features[i:i+seq_len])
            y_seqs.append(window_labels[0])
            i += stride
        else:
            # skip forward until label changes align
            i += 1
    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=object)
    return X, y

# -------------------------
# LOAD + PREPARE
# -------------------------
print("Loading data...")
df, labels, features = load_csv(CSV_PATH)
print(f"Total frames: {features.shape[0]}, feature dim: {features.shape[1]}")

print("Creating sequences...")
X, y = create_sequences(labels, features, seq_len=SEQ_LEN, stride=STRIDE)
print(f"Created sequences: {X.shape}, labels: {y.shape}")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

# Flatten frames to fit scaler, scale per-dimension
n_seq, s_len, n_feat = X.shape
X_reshaped = X.reshape(-1, n_feat)
scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled_flat.reshape(n_seq, s_len, n_feat)

# Train / test / val split (stratified by label)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc)

val_rel = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_rel, random_state=RANDOM_STATE, stratify=y_temp)

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# Save label encoder and scaler for inference
with open(os.path.join(MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({"classes": le.classes_.tolist()}, f)

import joblib
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))

# -------------------------
# DATASET (tf.data)
# -------------------------
def make_dataset(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds = make_dataset(X_val, y_val, shuffle=False)
test_ds = make_dataset(X_test, y_test, shuffle=False)

# -------------------------
# MODEL ARCHITECTURES
# -------------------------
def build_lstm(seq_len=SEQ_LEN, n_feat=n_feat, n_classes=num_classes, units=128, dropout=0.3):
    inp = layers.Input(shape=(seq_len, n_feat))
    x = layers.Masking()(inp)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(units//2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="LSTM_model")
    return model

def build_gru(seq_len=SEQ_LEN, n_feat=n_feat, n_classes=num_classes, units=128, dropout=0.3):
    inp = layers.Input(shape=(seq_len, n_feat))
    x = layers.Masking()(inp)
    x = layers.GRU(units, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GRU(units//2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="GRU_model")
    return model

def build_cnn1d(seq_len=SEQ_LEN, n_feat=n_feat, n_classes=num_classes):
    inp = layers.Input(shape=(seq_len, n_feat))
    # apply 1D conv across time for each feature vector
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out, name="CNN1D_model")
    return model

# -------------------------
# TRAIN FUNCTION
# -------------------------
def compile_and_train(model, model_name, train_ds, val_ds, epochs=EPOCHS):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    cb = [
        callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, f"{model_name}.h5"),
                                  monitor="val_accuracy",
                                  save_best_only=True, save_weights_only=False),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
    return history

# -------------------------
# TRAIN ALL OR ONE
# -------------------------
if __name__ == "__main__":
    # Choose which model(s) to train
    # 1) LSTM
    lstm = build_lstm()
    print("\nTraining LSTM model")
    compile_and_train(lstm, "lstm_cricket", train_ds, val_ds, epochs=EPOCHS)

    # 2) GRU
    gru = build_gru()
    print("\nTraining GRU model")
    compile_and_train(gru, "gru_cricket", train_ds, val_ds, epochs=EPOCHS)

    # 3) 1D-CNN
    cnn = build_cnn1d()
    print("\nTraining 1D-CNN model")
    compile_and_train(cnn, "cnn1d_cricket", train_ds, val_ds, epochs=EPOCHS)

    # Evaluate on test set (example: evaluate LSTM)
    print("Evaluating LSTM on test set:")
    print(lstm.evaluate(test_ds, verbose=1))

    # Save final chosen model (if needed)
    lstm.save(os.path.join(MODEL_DIR, "lstm_final.h5"))
    print("Models and scaler/label map saved to", MODEL_DIR)
