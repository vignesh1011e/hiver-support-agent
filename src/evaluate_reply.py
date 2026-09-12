import argparse
import json
import re

import pandas as pd

from agent import keyword_classify, retrieve, draft_reply


def contains_unsupported_guarantee(text):
    """
    Flag wording that could make unsupported promises.
    """
    patterns = [
        r"\bdefinitely\b",
        r"\bguaranteed\b",
        r"\bguarantee\b",
        r"\bwill arrive\b",
        r"\bwill be delivered\b",
        r"\bwill be refunded\b",
        r"\brefund will\b",
        r"\bwill receive\b",
    ]

    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def lexical_grounding(reply, historical_reply):
    """
    Measure how much of the drafted response overlaps
    with the retrieved historical response.

    This is a simple automated grounding proxy, not a semantic
    grounding score.
    """

    reply_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", reply.lower())
    )

    historical_words = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", historical_reply.lower())
    )

    if not reply_words:
        return 0.0

    overlap = reply_words.intersection(historical_words)

    return round(len(overlap) / len(reply_words), 4)


def clean_historical_reply(text):
    """
    Apply the same cleaning used by the agent so that
    grounding is measured against the actual customer-facing
    historical response.
    """

    text = str(text).strip()

    # Remove Twitter handles.
    text = re.sub(r"@\w+", "", text)

    # Remove agent signatures such as ^AM or ^KK.
    text = re.sub(r"\^[A-Z]{2}\b", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def evaluate_row(message, true_intent, customer_tweet_id, index):
    """
    Run the reply pipeline while excluding the Golden Set case itself
    from retrieval to prevent self-retrieval/data leakage.
    """

    intent, confidence = keyword_classify(message)

    # Retrieve historical examples.
    examples = retrieve(
        index,
        message
    )

    # Exclude the Golden Set's own historical case.
    if len(examples) > 0 and "customer_tweet_id" in examples.columns:
        examples = examples[
            examples["customer_tweet_id"].astype(str)
            != str(customer_tweet_id)
        ].copy()

    reply = draft_reply(
        message,
        intent,
        examples
    )

    if len(examples) > 0:
        best = examples.iloc[0]

        similarity = float(
            best["similarity"]
        )

        historical_reply = clean_historical_reply(
            best["amazon_response"]
        )

        grounding = lexical_grounding(
            reply,
            historical_reply
        )

    else:
        similarity = 0.0
        historical_reply = ""
        grounding = 0.0

    unsupported_guarantee = contains_unsupported_guarantee(
        reply
    )

    return {
        "customer_tweet_id": str(customer_tweet_id),
        "message": message,
        "true_intent": true_intent,
        "predicted_intent": intent,
        "confidence": round(float(confidence), 4),
        "reply": reply,
        "retrieval_similarity": round(similarity, 4),
        "grounding_overlap": grounding,
        "unsupported_guarantee": unsupported_guarantee,
        "has_reply": bool(reply.strip()),
        "retrieved_historical_reply": historical_reply,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--golden",
        default="data/golden_set.csv"
    )

    parser.add_argument(
        "--index",
        default="data/retrieval_index.joblib"
    )

    parser.add_argument(
        "--out",
        default="results/reply_results.json"
    )

    args = parser.parse_args()

    golden = pd.read_csv(
        args.golden
    )

    import joblib

    index = joblib.load(
        args.index
    )

    print(
        f"Evaluating {len(golden)} Golden Set examples..."
    )

    results = []

    for i, row in golden.iterrows():

        result = evaluate_row(
    str(row["customer_text"]),
    str(row["intent"]),
    str(row["customer_tweet_id"]),
    index
)

        results.append(result)

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{len(golden)}"
            )

    results_df = pd.DataFrame(results)

    summary = {
        "n_examples": len(results_df),

        "retrieval": {
            "mean_top_similarity": round(
                float(
                    results_df[
                        "retrieval_similarity"
                    ].mean()
                ),
                4
            ),

            "median_top_similarity": round(
                float(
                    results_df[
                        "retrieval_similarity"
                    ].median()
                ),
                4
            ),
        },

        "grounding": {
            "mean_lexical_overlap": round(
                float(
                    results_df[
                        "grounding_overlap"
                    ].mean()
                ),
                4
            )
        },

        "safety": {
            "unsupported_guarantee_rate": round(
                float(
                    results_df[
                        "unsupported_guarantee"
                    ].mean()
                ),
                4
            ),

            "empty_reply_rate": round(
                float(
                    (~results_df["has_reply"])
                    .mean()
                ),
                4
            ),
        },
    }

    output = {
        "summary": summary,
        "examples": results,
    }

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nReply Evaluation Results:")
    print(
        json.dumps(
            summary,
            indent=2
        )
    )

    print(
        f"\nSaved results to {args.out}"
    )


if __name__ == "__main__":
    main()