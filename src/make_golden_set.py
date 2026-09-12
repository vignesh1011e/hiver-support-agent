import argparse, pandas as pd

INTENTS = [
    'delivery_delay', 'delivery_missing', 'wrong_or_damaged_item',
    'order_change_cancel', 'return_refund', 'payment_billing',
    'prime_membership', 'account_login', 'product_device_issue',
    'general_information', 'other_unclear'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=['customer_text']).drop_duplicates('customer_tweet_id')
    n = min(args.n, len(df))
    # Deterministic random sample. Labels are intentionally blank: the Golden Set is hand-labelled.
    sample = df.sample(n=n, random_state=42)[['customer_tweet_id','customer_text','amazon_response']].copy()
    sample.insert(3, 'intent', '')
    sample.insert(4, 'should_escalate', '')
    sample.insert(5, 'label_notes', '')
    sample.to_csv(args.output, index=False)
    print(f'Created {len(sample)} unlabeled Golden Set rows: {args.output}')
    print('Manually fill intent, should_escalate, and optionally label_notes.')
    print('Allowed starting intents:')
    for x in INTENTS:
        print('  -', x)

if __name__ == '__main__':
    main()
