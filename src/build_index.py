import argparse, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.input).dropna(subset=['customer_text']).drop_duplicates('customer_tweet_id').reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df['customer_text'].astype(str))
    nn = NearestNeighbors(n_neighbors=min(5, len(df)), metric='cosine', algorithm='brute')
    nn.fit(matrix)
    joblib.dump({'df': df, 'vectorizer': vectorizer, 'matrix': matrix, 'nn': nn}, args.out)
    print(f'Indexed {len(df):,} cases.')

if __name__ == '__main__':
    main()
