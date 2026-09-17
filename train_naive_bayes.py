"""
train_naive_bayes.py
---------------------
Trains a Multinomial Naive Bayes model (TF-IDF features) for tweet
sentiment classification and saves the fitted vectorizer + model.
"""

import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from preprocess import prepare_dataset

TRAIN_PATH = "data/twitter_training.csv"
VAL_PATH = "data/twitter_validation.csv"
MODEL_DIR = "models"


def main():
    train_df, val_df = prepare_dataset(TRAIN_PATH, VAL_PATH)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"Multinomial Naive Bayes validation accuracy: {accuracy:.4f}")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    joblib.dump(model, f"{MODEL_DIR}/naive_bayes_model.joblib")
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.joblib")

    with open(f"{MODEL_DIR}/nb_accuracy.json", "w") as f:
        json.dump({"model": "Multinomial NB", "accuracy": accuracy}, f)


if __name__ == "__main__":
    main()
