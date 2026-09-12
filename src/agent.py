import argparse
import json
import joblib
import re


# -----------------------------
# Safety / escalation policy
# -----------------------------

RISKY_INTENTS = {
    "payment_billing",
    "return_refund",
    "account_login",
    "delivery_missing",
    "wrong_or_damaged_item",
}

CONFIDENCE_THRESHOLD = 0.70
EVIDENCE_THRESHOLD = 0.35


# -----------------------------
# Intent patterns
# -----------------------------

LABEL_PATTERNS = {
    "delivery_delay": [
        r"\blate\b",
        r"\bdelayed\b",
        r"\bdelay\b",
        r"\bstill waiting\b",
        r"\bwaiting\b",
        r"\bhasn't arrived\b",
        r"\bhasnt arrived\b",
        r"\bnot arrived\b",
        r"\bwhen will\b",
        r"\bwhen can i expect\b",
        r"\bdelivery date\b",
        r"\bdelivery time\b",
        r"\bexpected delivery\b",
        r"\bshipping\b",
        r"\bshipped\b",
        r"\bdispatch(ed)?\b",
        r"\bnot dispatched\b",
        r"\bnext day delivery\b",
        r"\bdelivery is late\b",
        r"\bwhere is my order\b",
        r"\bwhere's my order\b",
        r"\bwhere is my package\b",
        r"\bwhere's my package\b",
        r"\bupdate on (my )?(order|package|parcel)\b",
        r"\btimeline\b",
    ],

    "delivery_missing": [
        r"\bmarked as delivered\b",
        r"\bsays delivered\b",
        r"\bshows delivered\b",
        r"\bdelivered but\b",
        r"\bdelivered.*not received\b",
        r"\bnever received\b",
        r"\bdidn't receive\b",
        r"\bdidnt receive\b",
        r"\bnot received\b",
        r"\bnever got\b",
        r"\bdidn't get\b",
        r"\bdidnt get\b",
        r"\bpackage.*missing\b",
        r"\bparcel.*missing\b",
        r"\border.*missing\b",
        r"\bpackage is missing\b",
        r"\bparcel is missing\b",
        r"\blost package\b",
        r"\blost parcel\b",
        r"\bwhere is my package\b",
    ],

    "wrong_or_damaged_item": [
        r"\bwrong item\b",
        r"\bwrong product\b",
        r"\bdifferent item\b",
        r"\bincorrect item\b",
        r"\bwrong order\b",
        r"\bdamaged\b",
        r"\bdamage(d)?\b",
        r"\bbroken\b",
        r"\bdefective\b",
        r"\bfake product\b",
        r"\bcounterfeit\b",
        r"\bmissing item\b",
        r"\breceived.*instead\b",
        r"\bsent.*instead\b",
        r"\btwo left\b",
        r"\btwo right\b",
    ],

    "order_change_cancel": [
        r"\bcancel\b",
        r"\bcancelled\b",
        r"\bcanceled\b",
        r"\bcancellation\b",
        r"\bchange my order\b",
        r"\bchange order\b",
        r"\bmodify my order\b",
        r"\bmodify order\b",
        r"\bchange.*address\b",
        r"\bupdate.*address\b",
        r"\bchange.*delivery address\b",
    ],

    "return_refund": [
        r"\brefund\b",
        r"\brefunded\b",
        r"\brefund(ed|ing)?\b",
        r"\breturn\b",
        r"\breturned\b",
        r"\breturning\b",
        r"\breturn this\b",
        r"\bsend it back\b",
        r"\bsend.*back\b",
        r"\bmoney back\b",
        r"\breturn pickup\b",
        r"\bpickup.*return\b",
    ],

    "payment_billing": [
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bcharged twice\b",
        r"\bdouble charge\b",
        r"\bpayment\b",
        r"\bpay(ment)?\b",
        r"\bbilling\b",
        r"\bbill\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bbank account\b",
        r"\bbank statement\b",
        r"\bcard declined\b",
        r"\bdeclined my card\b",
        r"\btransaction\b",
        r"\bdisbursement\b",
        r"\bcashback\b",
        r"\bamount\b.*\bcharged\b",
        r"\bcharged\b.*\bamount\b",
        r"\bdeduct(ed|ion)?\b",
        r"\bmoney\b.*\btaken\b",
        r"\btook my money\b",
    ],

    "prime_membership": [
        r"\bamazon prime\b",
        r"\bprime membership\b",
        r"\bprime subscription\b",
        r"\bprime trial\b",
        r"\bfree prime\b",
        r"\brenew.*prime\b",
        r"\bprime.*renew\b",
        r"\bprime charge\b",
        r"\bprime fee\b",
        r"\bprime benefits\b",
        r"\bprime delivery\b",
        r"\bprime\b",
    ],

    "account_login": [
        r"\blogin\b",
        r"\blog in\b",
        r"\bsign in\b",
        r"\bsign-in\b",
        r"\bcan't access\b",
        r"\bcant access\b",
        r"\bcannot access\b",
        r"\baccess my account\b",
        r"\baccount locked\b",
        r"\blocked account\b",
        r"\bpassword\b",
        r"\baccount access\b",
        r"\bsuspended account\b",
        r"\bbanned account\b",
        r"\baccount on hold\b",
    ],

    "product_device_issue": [
        r"\bkindle\b",
        r"\becho\b",
        r"\balexa\b",
        r"\bfire tv\b",
        r"\bfire hd\b",
        r"\bdevice\b",
        r"\bapp\b",
        r"\bapplication\b",
        r"\bnot working\b",
        r"\bdoesn't work\b",
        r"\bdoesnt work\b",
        r"\bwon't work\b",
        r"\bwont work\b",
        r"\berror\b",
        r"\berror message\b",
        r"\bdownload\b",
        r"\bplayback\b",
        r"\binstallation\b",
        r"\binstall\b",
        r"\bcheckout\b",
        r"\bcan't order\b",
        r"\bcant order\b",
        r"\bunable to order\b",
        r"\btechnical issue\b",
        r"\bquality drops\b",
    ],

    "general_information": [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwhat does\b",
        r"\bhow do i\b",
        r"\bhow can i\b",
        r"\bwhere can i\b",
        r"\bwhere do i\b",
        r"\bwhere is the option\b",
        r"\bcan i\b",
        r"\bdo you offer\b",
        r"\bis there\b",
        r"\bis it available\b",
        r"\bavailable\b",
        r"\bavailability\b",
        r"\bwhich\b",
        r"\bwhen is\b",
        r"\bprice\b",
        r"\bpricing\b",
        r"\bspecs\b",
        r"\bspecifications\b",
        r"\bwarranty\b",
        r"\bguarantee\b",
        r"\bpolicy\b",
        r"\bpromotion\b",
        r"\bpromo\b",
        r"\boffer\b",
        r"\bdiscount\b",
        r"\bwebsite\b",
        r"\bhelp page\b",
        r"\bsupport email\b",
        r"\bemail address\b",
    ],
}


def normalize(text):
    """
    Normalize noisy Twitter text before classification.
    """
    text = str(text).lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter handles.
    text = re.sub(r"@\w+", " ", text)

    # Normalize HTML entities.
    text = text.replace("&amp;", " and ")

    # Normalize apostrophes.
    text = text.replace("’", "'")
    text = text.replace("‘", "'")

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def keyword_classify(message):
    """
    Deterministic rule-based classifier.

    Rules are intentionally transparent so the system can be
    explained and reproduced without an external API.
    """

    text = normalize(message)

    scores = {
        intent: 0
        for intent in LABEL_PATTERNS
    }

    matched = {
        intent: []
        for intent in LABEL_PATTERNS
    }

    for intent, patterns in LABEL_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, text):

                scores[intent] += 1
                matched[intent].append(pattern)

    # --------------------------------
    # Special precedence rules
    # --------------------------------

    # "Delivered but not received" is missing,
    # not merely delayed.
    if (
        re.search(r"\b(delivered|marked as delivered|shows delivered)\b", text)
        and re.search(
            r"\b(not received|never received|didn't receive|didnt receive|never got)\b",
            text,
        )
    ):
        scores["delivery_missing"] += 3

    # A clearly financial Prime problem should remain payment-related.
    if (
        "prime" in text
        and re.search(
            r"\b(charged|charge|payment|bank|card|money|deduct|billing)\b",
            text,
        )
    ):
        scores["payment_billing"] += 2

    # A Prime delivery complaint is still delivery unless
    # membership itself is the subject.
    if (
        "prime" in text
        and re.search(
            r"\b(delivery|delivered|late|delay|arrived|shipping|package|parcel)\b",
            text,
        )
        and not re.search(
            r"\b(membership|subscription|trial|renew|benefits|fee)\b",
            text,
        )
    ):
        scores["delivery_delay"] += 2

    # --------------------------------
    # Select best intent
    # --------------------------------

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    if best_score == 0:
        return "other_unclear", 0.30

    # Difference between first and second-best intent.
    ordered = sorted(
        scores.values(),
        reverse=True
    )

    second_score = ordered[1] if len(ordered) > 1 else 0

    total = sum(scores.values())

    # Confidence rewards:
    # - multiple matching phrases
    # - separation from second-best intent
    confidence = (
        0.55
        + min(best_score / 4, 1.0) * 0.25
        + min(
            (best_score - second_score) / max(best_score, 1),
            1.0
        ) * 0.18
    )

    confidence = min(
        0.98,
        max(0.30, confidence)
    )

    # If several intents have similar scores, lower confidence.
    if second_score > 0 and best_score == second_score:
        confidence = min(confidence, 0.60)

    return best_intent, confidence


def retrieve(index, message, k=5):

    query = index["vectorizer"].transform(
        [message]
    )

    distances, ids = index["nn"].kneighbors(
        query,
        n_neighbors=min(
            k,
            len(index["df"])
        )
    )

    rows = index["df"].iloc[ids[0]].copy()

    rows["similarity"] = (
        1 - distances[0]
    )

    return rows


def draft_reply(message, intent, examples):
    """
    Grounded response generation.

    The response comes from historical AmazonHelp resolutions.
    We do not invent policy or guarantees.
    """

    if len(examples) == 0:
        return (
            "Thanks for reaching out. "
            "We need to review your issue further "
            "before providing a resolution."
        )

    best = examples.iloc[0]

    historical_reply = str(
        best["amazon_response"]
    ).strip()

    # Remove Twitter handles.
    historical_reply = re.sub(
        r"@\w+",
        "",
        historical_reply
    )

    # Remove agent signature markers such as ^AM or ^KK.
    historical_reply = re.sub(
        r"\^[A-Z]{2}\b",
        "",
        historical_reply
    )

    # Remove excessive whitespace.
    historical_reply = re.sub(
        r"\s+",
        " ",
        historical_reply
    ).strip()

    if not historical_reply:
        return (
            "Thanks for reaching out. "
            "Please contact Amazon customer support "
            "so we can look into this further."
        )

    return historical_reply
    
def decide(
    intent,
    confidence,
    examples,
    draft_text,
    message="",
):
    """
    Conservative but intent-aware automation policy.

    Escalate when:
      - the intent is inherently risky
      - intent confidence is low
      - the customer is asking for account/order-specific action
      - the message contains a strong unresolved-problem signal

    Retrieval similarity is supporting evidence, not a hard safety gate.
    """

    text = normalize(message)

    reasons = []


        # Simple acknowledgements can be safely handled automatically.
    acknowledgement_patterns = [
    r"^\s*thank you\s*[.!]?\s*$",
    r"^\s*thanks\s*[.!]?\s*$",
    r"^\s*thank you for .*\bhelp(?:ing)?\b.*[.!]?\s*$",
    r"^\s*thanks for .*\bhelp(?:ing)?\b.*[.!]?\s*$",
    r"^\s*you'?re welcome\s*[.!]?\s*$",
    r"^\s*you are welcome\s*[.!]?\s*$",
]

    if any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in acknowledgement_patterns
    ):
        return (
            "AUTO_HANDLE",
            "simple acknowledgement with no unresolved support request",
        )

    # --------------------------------
    # 1. High-risk intents
    # --------------------------------

    if intent in RISKY_INTENTS:
        reasons.append(
            f"{intent} requires human review"
        )

    # --------------------------------
    # 2. Low classification confidence
    # --------------------------------

    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"low intent confidence ({confidence:.2f})"
        )

    # --------------------------------
    # 3. Strong unresolved-support signals
    # --------------------------------

    resolution_patterns = [
        r"\bplease help\b",
        r"\bcan you help\b",
        r"\bhelp me\b",
        r"\bwhat should i do\b",
        r"\bwhat can i do\b",
        r"\bwhat do i do\b",
        r"\bneed help\b",
        r"\bneed assistance\b",
        r"\bcontact\b",
        r"\bcall\b",
        r"\bphone number\b",
        r"\bcustomer service\b",
        r"\bsupport\b",
        r"\bcomplaint\b",
        r"\bcomplaining\b",
        r"\brefund\b",
        r"\bcharged\b",
        r"\bpayment\b",
        r"\baccount\b",
        r"\bcan't access\b",
        r"\bcant access\b",
        r"\bnot received\b",
        r"\bnever received\b",
        r"\bmissing\b",
        r"\bdamaged\b",
        r"\bwrong item\b",
        r"\bcancel\b",
    ]

    resolution_signal = any(
        re.search(pattern, text)
        for pattern in resolution_patterns
    )

    if resolution_signal and intent not in {
        "general_information",
        "other_unclear",
    }:
        reasons.append(
            "message contains an unresolved support/action signal"
        )

    # --------------------------------
    # 4. Very uncertain general/unclear cases
    # --------------------------------

    if (
        intent == "other_unclear"
        and confidence < 0.75
    ):
        reasons.append(
            "customer intent is unclear"
        )

    # --------------------------------
    # 5. Unsupported guarantees
    # --------------------------------

    if re.search(
        r"\b(definitely|guaranteed|"
        r"will arrive|will be refunded|"
        r"guarantee)\b",
        draft_text,
        re.IGNORECASE,
    ):
        reasons.append(
            "draft contains a potentially unsupported guarantee"
        )

    # --------------------------------
    # Decision
    # --------------------------------

    if reasons:
        return (
            "ESCALATE",
            "; ".join(reasons)
        )

    return (
        "AUTO_HANDLE",
        "low-risk request with sufficient confidence "
        "and no unresolved support trigger",
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message",
        required=True,
        help="Incoming customer message",
    )

    parser.add_argument(
        "--index",
        default="data/retrieval_index.joblib",
        help="Retrieval index",
    )

    args = parser.parse_args()

    index = joblib.load(
        args.index
    )

    examples = retrieve(
        index,
        args.message,
        k=5,
    )

    intent, confidence = keyword_classify(
        args.message
    )

    reply = draft_reply(
        args.message,
        intent,
        examples,
    )

    decision, reason = decide(
    intent,
    confidence,
    examples,
    reply,
    args.message,
)

    result = {
        "intent": intent,
        "confidence": round(
            confidence,
            3
        ),
        "reply": reply,
        "decision": decision,
        "reason": reason,
        "evidence": examples[
            [
                "customer_text",
                "amazon_response",
                "similarity",
            ]
        ].to_dict(
            "records"
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()