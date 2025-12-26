import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from features import FeatureExtractor

def load_data(filepath):
    """
    Loads data from CSV.
    Expected format: CSV with 'url' and 'status' (or 'label') columns.
    Adjust column names as per dataset.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Basic cleaning - adjust based on actual dataset inspection
    # Common column names in phishing datasets: 'url', 'phishing', 'status', 'label'
    
    # Example for typical datasets:
    # Updated Logic: Treat 'benign' as 0, everything else (phishing, defacement, malware) as 1
    if 'status' in df.columns:
        df['label'] = df['status'].apply(lambda x: 0 if x == 'benign' else 1)
    elif 'type' in df.columns:
        df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    
    return df

def extract_features_from_df(df):
    """
    Apply FeatureExtractor to every URL in the dataframe.
    """
    print("Extracting features (this may take a while)...")
    
    features_list = []
    labels = []
    
    for index, row in df.iterrows():
        try:
            url = row['url']
            extractor = FeatureExtractor(url)
            features = extractor.get_features()
            features_list.append(features)
            labels.append(row['label'])
        except Exception as e:
            # Skip malformed URLs
            continue
            
    return pd.DataFrame(features_list), labels

def train_model(X, y):
    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    return rf

def save_model(model, filepath='model.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phishing URL Detector")
    parser.add_argument('dataset', help="Path to the training CSV")
    args = parser.parse_args()
    
    df = load_data(args.dataset)
    X, y = extract_features_from_df(df)
    model = train_model(X, y)
    save_model(model)
