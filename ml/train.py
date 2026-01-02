import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from features import FeatureExtractor

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

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

def extract_features_from_df(df, cache_path='ml/data/features_cache.csv'):
    """
    Apply FeatureExtractor to every URL in the dataframe.
    Uses caching to speed up subsequent runs.
    """
    if os.path.exists(cache_path):
        print(f"Loading features from cache: {cache_path}")
        cached_df = pd.read_csv(cache_path)
        # Ensure labels are present
        if 'label' not in cached_df.columns:
             print("Cache invalid (missing labels). Regenerating...")
        else:
             # Drop non-feature columns for X
             y = cached_df['label']
             X = cached_df.drop(columns=['label', 'url'], errors='ignore')
             X = X.reindex(sorted(X.columns), axis=1)
             return X, y
        
    print("Extracting features (this may take a while)...")
    
    features_list = []
    labels = []
    valid_urls = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        try:
            url = row['url']
            extractor = FeatureExtractor(url)
            features = extractor.get_features()
            features_list.append(features)
            labels.append(row['label'])
            valid_urls.append(url)
        except Exception as e:
            # Skip malformed URLs
            continue
    
    X = pd.DataFrame(features_list)
    # Enforce alphabetical order for consistency
    X = X.reindex(sorted(X.columns), axis=1)

    # Add label and url for caching
    cache_df = X.copy()
    cache_df['label'] = labels
    cache_df['url'] = valid_urls
    
    # Save cache
    print(f"Saving features to cache: {cache_path}")
    cache_df.to_csv(cache_path, index=False)
    
    return X, labels

def train_model(X, y):
    print("Training Random Forest model...")
    # Stratified split ensures equal proportion of classes in train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    return rf

def save_model(model, filepath=None):
    if filepath is None:
        # Default to saving in the same directory as this script
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
        
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phishing URL Detector")
    parser.add_argument('dataset', help="Path to the training CSV")
    parser.add_argument('--out', help="Path to save the model", default=None)
    args = parser.parse_args()
    
    df = load_data(args.dataset)
    X, y = extract_features_from_df(df)
    model = train_model(X, y)
    save_model(model, args.out)
