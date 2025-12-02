import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import argparse

def load_logs(log_file):
    """Load log file into a DataFrame."""
    with open(log_file, "r") as f:
        logs = [line.strip() for line in f.readlines()]
    return pd.DataFrame({"log": logs})

def vectorize_logs(logs_df):
    """Convert text logs into numerical features using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(logs_df["log"])
    return vectors, vectorizer

def detect_anomalies(vectors):
    """Run Isolation Forest for anomaly detection."""
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,  # 2% anomalies expected
        random_state=42
    )
    model.fit(vectors)
    preds = model.predict(vectors)  # -1 = anomaly, 1 = normal
    return preds, model

def save_results(logs_df, preds, output_file):
    """Save anomaly results."""
    logs_df["anomaly"] = preds
    logs_df["anomaly"] = logs_df["anomaly"].apply(lambda x: "YES" if x == -1 else "NO")
    logs_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Log anomaly detection using Isolation Forest")
    parser.add_argument("--input", required=True, help="Path to log file")
    parser.add_argument("--output", default="anomaly_results.csv", help="Output CSV file")
    args = parser.parse_args()

    print("Loading logs...")
    logs_df = load_logs(args.input)
    
    print("Vectorizing logs...")
    vectors, vectorizer = vectorize_logs(logs_df)
    
    print("Detecting anomalies...")
    preds, model = detect_anomalies(vectors)
    
    print("Saving results...")
    save_results(logs_df, preds, args.output)
    
    print("Completed successfully!")

if __name__ == "__main__":
    main()
