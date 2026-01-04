import pandas as pd
import numpy as np
import pickle
import os
import argparse
import sys
from urllib.parse import urlsplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from features import FeatureExtractor
from tqdm import tqdm
from xgboost import XGBClassifier

# Imports for CharCNN
try:
    import tensorflow as tf
    from char_cnn import CharCNN
except ImportError:
    print("TensorFlow not installed.")
    sys.exit(1)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Drop common index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Standardize label column to 0 (safe) and 1 (unsafe)
    if 'status' in df.columns:
        df['label'] = df['status'].astype(str).str.lower().apply(lambda x: 0 if x == 'benign' else 1).astype(int)
    elif 'type' in df.columns:
        df['label'] = df['type'].astype(str).str.lower().apply(lambda x: 0 if x == 'benign' else 1).astype(int)
    elif "result" in df.columns:
        df["label"] = df["result"].astype(int)
    elif "label" in df.columns:
        # allow datasets that already have label column
        if df["label"].dtype == object:
            df["label"] = df["label"].astype(str).str.lower().apply(lambda x: 0 if x == 'benign' else 1).astype(int)
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise ValueError("Dataset must contain one of: result, status, type, label.")
     
    return df

def get_hostname(u):
    try:
        u = str(u)
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        return urlsplit(u).netloc.lower()
    except ValueError:
        return ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="Path to dataset2.csv")
    parser.add_argument('--lexical-model', default="ml/model.pkl")
    parser.add_argument('--cnn-model', default="ml/runs/char_cnn")
    args = parser.parse_args()

    # 1. Recreate the Split
    df = load_data(args.dataset)
    df["hostname"] = df["url"].apply(get_hostname)
    df = df[df["hostname"] != ""].reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(df, y=df["label"], groups=df["hostname"]))
    
    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_test = df_test["label"].astype(int).values
    urls_test = df_test["url"].astype(str).tolist()

    print(f"Test Set: {len(df_test)} samples")

    # 2. Load Models
    print("Loading Lexical Model...")
    with open(args.lexical_model, "rb") as f:
        lex_model = pickle.load(f)
    
    print("Loading Char-CNN...")
    cnn = CharCNN()
    cnn.load(args.cnn_model)

    # 3. Get Lexical Predictions
    # We need to extract features for the test set again (or load from cache if we want to be fancy, but let's just extract to be safe/simple for now)
    # Actually, to save time, let's try to load the cache if it exists, matching the validation logic in train.py
    # But for a benchmark script, caching logic duplication is annoying. 
    # Let's inspect if `ml/data/features_test_cache.csv` aligns with our `df_test`.
    # train.py saves it. If we use the same split logic, it should align.
    
    cache_path = "ml/data/features_test_cache.csv"
    X_test_lex = None
    
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        cached = pd.read_csv(cache_path)
        # Verify alignment
        if len(cached) == len(df_test) and cached["url"].iloc[0] == df_test["url"].iloc[0]:
            X_test_lex = cached.drop(columns=["url", "label"])
            print("Cache aligned directly.")
        else:
            print("Cache not aligned. We must generate features (this might take time)...")
    
    if X_test_lex is None:
        # Fallback generation
        rows = []
        for url in tqdm(urls_test, desc="Extracting Features"):
            try:
                rows.append(FeatureExtractor(url).get_features())
            except:
                rows.append({}) # Should not happen if training worked
        X_test_lex = pd.DataFrame(rows)
        # Reindex features
        if hasattr(lex_model, "feature_names_in_"):
            X_test_lex = X_test_lex.reindex(columns=lex_model.feature_names_in_, fill_value=0)

    # Predict Lexical (Probabilities)
    print("Running Lexical Inference...")
    prob_lex = lex_model.predict_proba(X_test_lex)[:, 1]

    # Predict CNN
    print("Running CNN Inference...")
    prob_cnn = cnn.predict(urls_test) # Returns flattened array of float

    # 4. Ensemble
    print("Calculating Ensemble Scores (0.4 Lex + 0.6 CNN)...")
    prob_ensemble = (0.4 * prob_lex) + (0.6 * prob_cnn)
    
    # Threshold at 0.5 for binary metric calculation? 
    # Or should we respect our 3-level policy?
    # For standard metrics (Accuracy/F1), we usually use 0.5 cutoff.
    # But let's seeing how many fall into "Warning" vs "Unsafe".
    
    y_pred = (prob_ensemble >= 0.5).astype(int)

    # 5. Report
    print("\n--- Hybrid Ensemble Performance ---")
    print(classification_report(y_test, y_pred))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Analysis of False Positives
    print("\n--- FALSE POSITIVES (Safe sites marked as Phishing) ---")
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    for i in fp_indices:
        url = urls_test[i]
        p_lex = prob_lex[i]
        p_cnn = prob_cnn[i]
        p_final = prob_ensemble[i]
        print(f"URL: {url}")
        print(f"  Lexical: {p_lex:.4f} | CNN: {p_cnn:.4f} | Final: {p_final:.4f}")
        print("-" * 50)
        
    # Analysis of False Negatives (Optional, top 5)
    print("\n--- FALSE NEGATIVES (Phishing sites missed) [Top 10] ---")
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    for i in fn_indices[:10]:
        url = urls_test[i]
        p_lex = prob_lex[i]
        p_cnn = prob_cnn[i]
        p_final = prob_ensemble[i]
        print(f"URL: {url}")
        print(f"  Lexical: {p_lex:.4f} | CNN: {p_cnn:.4f} | Final: {p_final:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
