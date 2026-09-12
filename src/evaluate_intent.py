import argparse
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.golden)

    df = df.dropna(subset=["customer_text", "intent"]).copy()

    X = df["customer_text"].astype(str)
    y = df["intent"].astype(str)

    print(f"Total labelled examples: {len(df)}")
    print("\nIntent distribution:")
    print(y.value_counts().to_string())

    # Stratified cross-validation keeps every fold reasonably balanced.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # -------------------------
    # Baseline 1: Majority class
    # -------------------------
    majority_class = y.value_counts().idxmax()
    majority_pred = [majority_class] * len(y)

    majority_accuracy = accuracy_score(y, majority_pred)
    majority_macro_f1 = f1_score(
        y,
        majority_pred,
        average="macro",
        zero_division=0
    )

    # -------------------------
    # Baseline 2: TF-IDF + Logistic Regression
    # -------------------------
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict"
    )

    tfidf_accuracy = accuracy_score(y, predictions)
    tfidf_macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    results = {
        "n_examples": len(df),
        "evaluation": "5-fold stratified cross-validation",
        "majority_class": majority_class,
        "majority_accuracy": round(float(majority_accuracy), 4),
        "majority_macro_f1": round(float(majority_macro_f1), 4),
        "tfidf_accuracy": round(float(tfidf_accuracy), 4),
        "tfidf_macro_f1": round(float(tfidf_macro_f1), 4)
    }

    print("\nResults:")
    print(json.dumps(results, indent=2))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()