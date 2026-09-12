import argparse, csv, os
from collections import defaultdict


def split_ids(value):
    if not value:
        return []
    return [x.strip() for x in str(value).replace(';', ',').split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-cases', type=int, default=10000)
    args = ap.parse_args()

    amazon_parent_ids = set()
    amazon_rows = {}
    print('Pass 1/2: finding AmazonHelp responses...')
    with open(args.input, 'r', encoding='utf-8', errors='replace', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('author_id') == 'AmazonHelp' and row.get('in_response_to_tweet_id'):
                for pid in split_ids(row['in_response_to_tweet_id']):
                    amazon_parent_ids.add(pid)
                amazon_rows[row['tweet_id']] = row

    print(f'Found {len(amazon_parent_ids):,} customer tweet IDs directly answered by AmazonHelp.')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    count = 0
    print('Pass 2/2: extracting customer + Amazon response pairs...')
    with open(args.input, 'r', encoding='utf-8', errors='replace', newline='') as f, open(args.output, 'w', encoding='utf-8', newline='') as out:
        reader = csv.DictReader(f)
        fields = ['customer_tweet_id','customer_text','created_at','amazon_response','amazon_tweet_id']
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            if row.get('tweet_id') not in amazon_parent_ids:
                continue
            if str(row.get('inbound')).lower() != 'true':
                continue
            response_ids = split_ids(row.get('response_tweet_id'))
            responses = [amazon_rows[rid] for rid in response_ids if rid in amazon_rows]
            if not responses:
                # The response ID can be absent/partial in noisy data; skip because we want verified evidence.
                continue
            for response in responses:
                writer.writerow({
                    'customer_tweet_id': row.get('tweet_id',''),
                    'customer_text': row.get('text',''),
                    'created_at': row.get('created_at',''),
                    'amazon_response': response.get('text',''),
                    'amazon_tweet_id': response.get('tweet_id',''),
                })
                count += 1
                if count >= args.max_cases:
                    print(f'Wrote {count:,} cases to {args.output}')
                    return
    print(f'Wrote {count:,} cases to {args.output}')


if __name__ == '__main__':
    main()
