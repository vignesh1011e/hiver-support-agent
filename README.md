Hiver Support Agent

A lightweight AI-assisted customer support agent built for the Hiver SDE Intern take-home assignment.

The system:

Classifies customer messages into data-derived support intents.

Retrieves historically similar AmazonHelp resolutions.

Drafts a response grounded in those historical resolutions.

Decides whether to auto-handle or escalate to a human.

The implementation favors grounded responses, conservative automation, and reproducible evaluation.

1. Approach

I selected AmazonHelp from the Customer Support on Twitter dataset because it provides a large and diverse set of customer-support interactions.

Pipeline:

Customer message
      |
      v
Intent classification
      |
      v
Historical AmazonHelp retrieval
      |
      v
Grounded response draft
      |
      v
Safety / escalation decision
      |
      +------------------+
      |                  |
      v                  v
 AUTO_HANDLE         ESCALATE

The response generator uses historical AmazonHelp resolutions rather than unconstrained generation, reducing the risk of inventing policies, refund amounts, delivery dates, or guarantees.

2. Dataset & Intent Taxonomy

Source: Customer Support on Twitter dataset by thoughtvector.

For development, approximately 10,000 AmazonHelp support cases were extracted from the original dataset.

The final taxonomy contains 11 intents:

delivery_delay

delivery_missing

wrong_or_damaged_item

order_change_cancel

return_refund

payment_billing

prime_membership

account_login

product_device_issue

general_information

other_unclear

A manually labelled 200-example Golden Set was used for evaluation and frozen before final testing.

3. Results

Intent classification

Two simple baselines were evaluated using 5-fold stratified cross-validation.

Approach

Accuracy

Macro-F1

Majority class

29.0%

4.09%

TF-IDF + Logistic Regression

32.5%

19.41%

Final rule-based agent

51.5%

49.92%

The final agent substantially outperformed both baselines.

The Golden Set is imbalanced and the smallest class contains four examples, so the corresponding cross-validation warning is reported as a limitation.

Escalation

Metric

Result

Accuracy

62.5%

Precision

76.92%

Recall

72.37%

F1

74.58%

Predicted auto-handle rate

28.5%

The escalation policy is intentionally conservative. Payment, account, refund, missing-delivery, and damaged/wrong-item cases are generally routed to humans.

4. Reply Generation & Evaluation

For each customer message, the system retrieves historically similar AmazonHelp cases and uses the corresponding historical support response as the draft.

Twitter handles and agent signature markers are removed before presenting the response.

Leakage-controlled evaluation

An initial evaluation produced a suspiciously high:

Mean retrieval similarity: 0.985

Median similarity: 1.000

Investigation showed that Golden Set examples could retrieve their own historical cases.

The evaluated case was therefore excluded from retrieval.

After leakage control:

Metric

Result

Mean top retrieval similarity

0.3269

Median top retrieval similarity

0.2537

Historical-response lexical grounding overlap

97.0%

Unsupported guarantee rate

1.0%

Empty reply rate

0%

The 97% figure is a lexical grounding proxy, not a claim that 97% of responses are high quality.

The leakage-controlled retrieval numbers are the meaningful generalization results.

5. What Is Misleading About My Headline Number?

The original 0.985 retrieval similarity is misleading because it was inflated by self-retrieval/data leakage: Golden Set examples were also present in the historical retrieval index.

After excluding each evaluated example's own historical case, similarity dropped to:

0.3269 mean / 0.2537 median

This is the more meaningful measure of retrieval generalization.

6. Top 5 Failure Modes

1. other_unclear is too broad

Many real support requests are incorrectly classified as other_unclear, particularly general-information, payment, delivery, and device-related cases.

2. General information vs. product/device

Generic product questions can resemble actual device problems, causing confusion between these two intents.

3. Delivery delay vs. delivery missing

Terms such as "still waiting", "late", and "not received" often overlap between delayed and missing deliveries.

4. Prime membership vs. payment/billing

Prime-related messages frequently contain financial language such as charges, payment, and subscription.

5. Short/noisy Twitter messages

Mentions, URLs, fragmented language, spelling variations, and insufficient context make some tweets difficult to classify reliably.

7. Evaluation & LLM-as-Judge

The repository contains automated evaluation for:

intent classification,

escalation decisions,

retrieval quality,

grounding,

unsupported guarantees.

An optional LLM-as-judge evaluator is implemented in src/llm_judge.py.

It scores:

relevance,

groundedness,

helpfulness,

safety,

overall quality.

The development API environment did not have usable API quota, so no unverified LLM-judge score is reported.

The Golden Set was manually labelled once. A second independent human annotation pass was not completed within the implementation window, so no fabricated human-agreement statistic is reported.

For a production evaluation, I would independently re-label a stratified subset and report Cohen's kappa for intent plus escalation agreement.

8. Reproduction

Python 3.11+ is recommended.

Install dependencies:

pip install -r requirements.txt

Build the retrieval index:

python src/build_index.py --input data/amazon_cases.csv --out data/retrieval_index.joblib

Run intent evaluation:

python src/evaluate_intent.py --golden data/golden_set.csv --out results/intent_results.json

Run full agent evaluation:

python src/evaluate_agent.py --golden data/golden_set.csv --out results/agent_results.json

Run reply evaluation:

python src/evaluate_reply.py --golden data/golden_set.csv --out results/reply_results.json

Test a single message:

python src/agent.py --message "My order is late and I am still waiting. Please help"

The original full dataset is not required for the evaluation above.

9. CI / Reproducibility

GitHub Actions runs the evaluation pipeline on pushes and pull requests.

Install dependencies
        |
        v
Build retrieval index
        |
        v
Intent evaluation
        |
        v
Agent evaluation
        |
        v
Reply evaluation
        |
        v
PASS / FAIL

The pipeline has been successfully executed on GitHub Actions.

10. Key Design Decisions

AmazonHelp selected for its large and diverse support interactions.

Data-derived intent taxonomy used instead of imposing a generic taxonomy.

Golden Set frozen before final evaluation.

other_unclear retained rather than forcing ambiguous messages into incorrect intents.

Majority and TF-IDF baselines included for comparison.

Historical retrieval used for grounding rather than unconstrained generation.

Retrieval similarity treated as evidence, not probability.

Sensitive intents escalated conservatively.

Simple acknowledgements can be auto-handled.

Self-retrieval explicitly tested and removed from evaluation.

No unsupported LLM-judge results reported.

CI uses the small reproducible evaluation set rather than the full multi-million-row dataset.

11. What I Would Do Next Week

Expand the Golden Set, especially underrepresented intents.

Add a second independent annotator and measure agreement.

Replace keyword rules with a supervised classifier trained on expanded labels.

Improve delivery-delay vs. missing-delivery disambiguation.

Add intent-aware retrieval and semantic reranking.

Run LLM-as-judge with a provisioned evaluation model.

Calibrate escalation thresholds against the cost of false automation.

Evaluate on a temporal holdout to measure robustness to changing support language.

Repository Structure

hiver_support_agent/
├── .github/
│   └── workflows/
│       └── evaluation.yml
├── data/
│   ├── amazon_cases.csv
│   ├── golden_set.csv
│   └── golden_set_template.csv
├── results/
├── src/
│   ├── agent.py
│   ├── build_index.py
│   ├── evaluate_agent.py
│   ├── evaluate_intent.py
│   ├── evaluate_reply.py
│   ├── extract_amazon_cases.py
│   ├── llm_judge.py
│   └── make_golden_set.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

Summary

The resulting system implements:

classify -> retrieve -> draft -> decide

On the frozen 200-example Golden Set:

51.5% intent accuracy / 49.92% macro-F1

74.58% escalation F1

28.5% predicted auto-handle rate

0% empty replies

1% unsupported-guarantee rate

0.3269 leakage-controlled mean retrieval similarity

97% historical-response lexical grounding overlap

The most important evaluation lesson was that the initial retrieval score was inflated by self-retrieval. Explicitly testing for and removing that leakage produced a more realistic estimate of retrieval generalization.

The implementation prioritizes grounded responses, conservative automation, transparent evidence, and reproducible evaluation.