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

from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit

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
    # if 'status' in df.columns:
    #     df['label'] = df['status'].apply(lambda x: 0 if x == 'benign' else 1)
    # elif 'type' in df.columns:
    #     df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)
    if "result" in df.columns:
        df["label"] = df["result"].astype(int)
        return df     
    
    return df

def extract_features_from_df(df, cache_path='ml/data/features_cache.csv'):
    """
    Apply FeatureExtractor to every URL in the dataframe.
    Uses caching to speed up subsequent runs.
    """
    if os.path.exists(cache_path):
        print(f"Loading features from cache: {cache_path}")
        return pd.read_csv(cache_path), df['label'] # Assuming aligned, but for safety usually better to save XY together.
        # Ideally we save the whole processed DF. For now let's regenerate X but check cache logic carefully.
        # Actually simplest: if cache exists, load X from it. But we need y. 
        # Let's save X and y together in cache.
        
    print("Extracting features (this may take a while)...")
    
    features_list = []
    labels = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        try:
            url = row['url']
            extractor = FeatureExtractor(url)
            features = extractor.get_features()
            features_list.append(features)
            labels.append(row['label'])
        except Exception as e:
            # Skip malformed URLs
            continue
    
    X = pd.DataFrame(features_list)
    
    # Save cache
    print(f"Saving features to cache: {cache_path}")
    X.to_csv(cache_path, index=False)
    
    return X, labels

def train_model(X_train, y_train, X_test, y_test):
    print("Training Random Forest model...")
    # Stratified split ensures equal proportion of classes in train/test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
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
    df["hostname"] = df["url"].astype(str).apply(lambda u: urlsplit(u if u.startswith(("http://","https://")) else "http://" + u).netloc.lower())
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y=df["label"], groups=df["hostname"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    X_train, y_train = extract_features_from_df(df_train, cache_path="ml/data/features_train_cache.csv")
    X_test,  y_test  = extract_features_from_df(df_test,  cache_path="ml/data/features_test_cache.csv")

    print(f"Train rows: {len(df_train)} | Test rows: {len(df_test)}")
    print(f"Unique hostnames train: {df_train['hostname'].nunique()} | test: {df_test['hostname'].nunique()}")


    X, y = extract_features_from_df(df)
    model = train_model(X_train, y_train, X_test, y_test)
    save_model(model, args.out)
