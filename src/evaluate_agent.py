import argparse
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def run_agent(message):
    """
    Local version of the support agent.

    This imports the same classification/retrieval logic
    used by src/agent.py without requiring an API.
    """
    from agent import (
        keyword_classify,
        retrieve,
        draft_reply,
        decide,
    )

    intent, confidence = keyword_classify(message)

    # Load retrieval index.
    import joblib

    index = joblib.load("data/retrieval_index.joblib")

    examples = retrieve(
        index,
        message,
        k=5
    )

    reply = draft_reply(
        message,
        intent,
        examples
    )

    decision, reason = decide(
    intent,
    confidence,
    examples,
    reply,
    message,
)

    return {
        "intent": intent,
        "confidence": confidence,
        "reply": reply,
        "decision": decision,
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--golden",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    args = parser.parse_args()

    df = pd.read_csv(args.golden)

    predictions = []

    print(f"Evaluating {len(df)} Golden Set examples...\n")

    for i, row in df.iterrows():

        result = run_agent(
            str(row["customer_text"])
        )

        predictions.append(result)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    pred_intents = [
        x["intent"]
        for x in predictions
    ]

    true_intents = (
        df["intent"]
        .astype(str)
        .tolist()
    )

    pred_escalation = [
        x["decision"] == "ESCALATE"
        for x in predictions
    ]

    true_escalation = (
        df["should_escalate"]
        .astype(str)
        .str.upper()
        .eq("TRUE")
        .tolist()
    )

    # Intent metrics
    intent_accuracy = accuracy_score(
        true_intents,
        pred_intents
    )

    intent_macro_f1 = f1_score(
        true_intents,
        pred_intents,
        average="macro",
        zero_division=0
    )

    # Escalation metrics
    escalation_accuracy = accuracy_score(
        true_escalation,
        pred_escalation
    )

    escalation_precision = precision_score(
        true_escalation,
        pred_escalation,
        zero_division=0
    )

    escalation_recall = recall_score(
        true_escalation,
        pred_escalation,
        zero_division=0
    )

    escalation_f1 = f1_score(
        true_escalation,
        pred_escalation,
        zero_division=0
    )

    # How often the agent automatically handles a request.
    auto_rate = sum(
        not x for x in pred_escalation
    ) / len(pred_escalation)

    results = {
        "n_examples": len(df),

        "intent": {
            "accuracy": round(
                float(intent_accuracy),
                4
            ),
            "macro_f1": round(
                float(intent_macro_f1),
                4
            ),
        },

        "escalation": {
            "accuracy": round(
                float(escalation_accuracy),
                4
            ),
            "precision": round(
                float(escalation_precision),
                4
            ),
            "recall": round(
                float(escalation_recall),
                4
            ),
            "f1": round(
                float(escalation_f1),
                4
            ),
        },

        "predicted_auto_handle_rate": round(
            float(auto_rate),
            4
        ),

        "examples": []
    }

    # Keep detailed predictions for error analysis.
    for i, row in df.iterrows():

        results["examples"].append({
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_text": row["customer_text"],
            "true_intent": true_intents[i],
            "predicted_intent": pred_intents[i],
            "true_escalate": true_escalation[i],
            "predicted_escalate": pred_escalation[i],
            "confidence": predictions[i]["confidence"],
            "reply": predictions[i]["reply"],
            "reason": predictions[i]["reason"],
        })

    with open(
        args.out,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nResults:")
    print(
        json.dumps(
            {
                k: v
                for k, v in results.items()
                if k != "examples"
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()
