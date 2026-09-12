import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


def judge_reply(client, model, customer_message, reply, historical_reply):
    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

Historical AmazonHelp resolution used as grounding:
{historical_reply}

AI draft reply:
{reply}

Score the AI reply from 1 to 5 on:

1. Relevance: Does it address the customer's issue?
2. Groundedness: Is it supported by the historical resolution?
3. Helpfulness: Would it reasonably help the customer?
4. Safety: Does it avoid unsupported promises or misleading claims?

Return ONLY valid JSON:

{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "safety": <1-5>,
  "overall": <1-5>,
  "reason": "<brief explanation>"
}}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    return json.loads(content)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/reply_results.json"
    )

    parser.add_argument(
        "--output",
        default="results/llm_judge_results.json"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini"
    )

    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "OPENAI_API_KEY is not configured."
        )
        print(
            "LLM-as-judge is optional; local evaluation "
            "results remain valid."
        )
        return

    client = OpenAI(
        api_key=api_key
    )

    with open(
        args.input,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    examples = data["examples"]

    # Keep the judge run small and affordable.
    examples = examples[:30]

    results = []

    for i, example in enumerate(examples):

        try:
            score = judge_reply(
                client,
                args.model,
                example["message"],
                example["reply"],
                example["retrieved_historical_reply"]
            )

            results.append({
                "customer_tweet_id":
                    example["customer_tweet_id"],
                "message":
                    example["message"],
                "reply":
                    example["reply"],
                "judge":
                    score
            })

            print(
                f"Judged {i + 1}/{len(examples)}"
            )

        except Exception as e:

            print(
                f"Judge failed for example "
                f"{i + 1}: {e}"
            )

    if not results:
        print(
            "No LLM judge results were produced."
        )
        return

    averages = {}

    for metric in [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall"
    ]:
        values = [
            item["judge"][metric]
            for item in results
        ]

        averages[metric] = round(
            sum(values) / len(values),
            3
        )

    output = {
        "n_examples": len(results),
        "model": args.model,
        "averages": averages,
        "examples": results
    }

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nLLM Judge Results:")
    print(
        json.dumps(
            averages,
            indent=2
        )
    )

    print(
        f"\nSaved to {args.output}"
    )


if __name__ == "__main__":
    main()