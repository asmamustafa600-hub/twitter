"""
train_gru.py
-------------
Trains a GRU model for tweet sentiment classification and saves
the fitted tokenizer + model.
"""

import json
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from preprocess import prepare_dataset

TRAIN_PATH = "data/twitter_training.csv"
VAL_PATH = "data/twitter_validation.csv"
MODEL_DIR = "models"

VOCAB_SIZE = 20000
MAX_LEN = 40
EMBED_DIM = 128
GRU_UNITS = 128
EPOCHS = 15
BATCH_SIZE = 128


def main():
    train_df, val_df = prepare_dataset(TRAIN_PATH, VAL_PATH)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])
    num_classes = len(label_encoder.classes_)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(val_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        GRU(GRU_UNITS, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nGRU validation accuracy: {accuracy:.4f}")

    model.save(f"{MODEL_DIR}/gru_model.keras")
    joblib.dump(tokenizer, f"{MODEL_DIR}/gru_tokenizer.joblib")

    with open(f"{MODEL_DIR}/gru_accuracy.json", "w") as f:
        json.dump({"model": "GRU", "accuracy": accuracy}, f)


if __name__ == "__main__":
    main()
