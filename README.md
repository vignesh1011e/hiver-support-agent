# Hiver SDE Intern — AmazonHelp Support Agent

A small, reproducible prototype for the Hiver take-home assignment.

## What it does

1. Extracts customer messages that received an `AmazonHelp` response from the Customer Support on Twitter dataset.
2. Builds a hand-labelled Golden Set from sampled cases.
3. Trains two intent baselines: majority class and TF-IDF + logistic regression.
4. Trains the same TF-IDF classifier used as the practical intent model.
5. Retrieves historically similar AmazonHelp resolutions.
6. Generates a grounded reply with an LLM when `OPENAI_API_KEY` is configured.
7. Applies a deterministic safety/escalation policy: low confidence, weak evidence, risky intents, or unsupported claims are escalated.

The repository intentionally works on a subsample. It does not require processing all ~3M tweets for evaluation.

## Recommended environment

Use Python 3.11+ and VS Code. Microsoft recommends a project-specific virtual environment for Python projects in VS Code. See the official VS Code Python guide: https://code.visualstudio.com/docs/python/python-tutorial

## 1. Data setup

Download the Kaggle Customer Support on Twitter ZIP and extract it so the project looks like:

```text
hiver_support_agent/
  data/
    twcs.csv
```

Do not commit `twcs.csv` to GitHub.

The script only reads the CSV sequentially, so it does not load the 500+ MB file into RAM.

## 2. Install

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

and run commands with `.\.venv\Scripts\python.exe`.

## 3. Extract AmazonHelp cases

```powershell
python src/extract_amazon_cases.py --input data/twcs.csv --output data/amazon_cases.csv --max-cases 10000
```

This finds customer tweets directly answered by `AmazonHelp` and stores the Amazon response as historical evidence.

## 4. Create the Golden Set

```powershell
python src/make_golden_set.py --input data/amazon_cases.csv --output data/golden_set.csv --n 200
```

Open `data/golden_set.csv` in Excel/VS Code and manually fill these two columns:

- `intent`
- `should_escalate`

Use the intent taxonomy only as a starting point. Inspect the actual messages and adjust the taxonomy before freezing it. Do not claim these labels were generated automatically.

Suggested initial taxonomy:

- `delivery_delay`
- `delivery_missing`
- `wrong_or_damaged_item`
- `order_change_cancel`
- `return_refund`
- `payment_billing`
- `prime_membership`
- `account_login`
- `product_device_issue`
- `general_information`
- `other_unclear`

For `should_escalate`, use `1` for human and `0` for auto. Label based on whether a support agent would need account/order-specific investigation, or whether an incorrect automated answer could cause meaningful harm.

## 5. Train/evaluate baselines

After the Golden Set is labelled:

```powershell
python src/evaluate_intent.py --golden data/golden_set.csv --out results/intent_results.json
```

This reports majority-class and TF-IDF + logistic-regression performance.

## 6. Build retrieval index

```powershell
python src/build_index.py --input data/amazon_cases.csv --out data/retrieval_index.joblib
```

## 7. Run the agent

Without an API key, it runs retrieval + deterministic escalation and prints a placeholder response.

With OpenAI:

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python src/agent.py --message "My package says delivered but I never received it"
```

The implementation uses the OpenAI Responses API and reads `OPENAI_API_KEY` from the environment. Official quickstart: https://platform.openai.com/docs/quickstart/make-your-first-api-request

For the take-home, never commit your API key.

## Evaluation design

Keep the test set frozen and separate from retrieval/training data. Report:

- majority baseline accuracy / macro F1
- TF-IDF + logistic regression accuracy / macro F1
- final intent model metrics
- reply quality scores from an LLM judge
- human-vs-LLM judge agreement on a 30–50 example subset
- safe automation rate
- top five failure modes

A useful headline business metric is:

`safe automation rate = cases auto-handled with correct intent + grounded reply + no unsupported claims / all evaluated cases`

Do not optimize only for the highest automation rate. Unsafe automation should count as a failure.
