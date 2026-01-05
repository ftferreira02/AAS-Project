import pandas as pd

import pickle
import argparse
import os
import json
from datetime import datetime

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
try:
    from char_cnn import CharCNN
except ImportError:
    CharCNN = None

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from features import FeatureExtractor
from tqdm import tqdm
from urllib.parse import urlsplit

def load_data(filepath):
    """
    Loads data from CSV.
    Expected format: CSV with 'url' and 'status' (or 'label') columns.
    Adjust column names as per dataset.
    """
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
        return df
    elif "label" in df.columns:
        # allow datasets that already have label column
        if df["label"].dtype == object:
            df["label"] = df["label"].astype(str).str.lower().apply(lambda x: 0 if x == 'benign' else 1).astype(int)
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise ValueError("Dataset must contain one of: result, status, type, label.")
     
    
    return df

def extract_features_from_df(df, cache_path):
    """
    Extract features for df["url"] and cache them together with url + label.
    Returns:
      X: DataFrame of features (no url/label columns)
      y: Series aligned with X
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Ensure expected columns exist
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame must contain 'url' and 'label' columns.")

    # Normalize URL column to string (do not alter content beyond this)
    urls_in = df["url"].astype(str).tolist()

    # If cache exists, validate and load
    if os.path.exists(cache_path):
        print(f"Loading features from cache: {cache_path}")
        cached = pd.read_csv(cache_path)

        # Basic schema check
        if not {"url", "label"}.issubset(set(cached.columns)):
            print("Cache missing required columns. Regenerating...")
        else:
            cached_urls = cached["url"].astype(str).tolist()

            # Strong validation: same length and same order
            if len(cached_urls) == len(urls_in) and cached_urls == urls_in:
                y = cached["label"].astype(int)
                X = cached.drop(columns=["url", "label"])
                return X, y
            else:
                print("Cache mismatch (different rows/order). Regenerating...")

    # Otherwise generate
    print("Extracting features (this may take a while)...")
    rows = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        url = str(row["url"])
        label = int(row["label"])
        try:
            feats = FeatureExtractor(url).get_features()
            feats["url"] = url
            feats["label"] = label
            rows.append(feats)
        except Exception:
            # If you skip, you MUST skip url+label too (we do by not appending)
            continue

    if not rows:
        raise RuntimeError("No features extracted (all rows failed parsing?).")

    out = pd.DataFrame(rows)

    # IMPORTANT: keep stable column ordering: url, label, then sorted feature columns
    feature_cols = sorted([c for c in out.columns if c not in ("url", "label")])
    out = out[["url", "label"] + feature_cols]

    print(f"Saving features to cache: {cache_path}")
    out.to_csv(cache_path, index=False)

    y = out["label"].astype(int)
    X = out.drop(columns=["url", "label"])
    return X, y

def build_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "rf":
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

    if model_name == "xgb":
        # Recommended by research for higher accuracy on lexical features
        return XGBClassifier(
            n_estimators=150, 
            learning_rate=0.1, 
            max_depth=10, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=3, # Fix class imbalance (approx 3:1 ratio)
            random_state=42,
            n_jobs=-1
        )

    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            ))
        ])

    if model_name == "rf_calibrated":
        base = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)

    if model_name == "xgb_calibrated":
        base = XGBClassifier(
            n_estimators=150, 
            learning_rate=0.1, 
            max_depth=10, 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=3, # Fix class imbalance
            random_state=42,
            n_jobs=-1
        )
        return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)

    raise ValueError(f"Unknown model '{model_name}'. Use: rf | xgb | logreg | rf_calibrated | xgb_calibrated")

def train_model(X_train, y_train, X_test, y_test, model_name="rf"):
    # Stratified split ensures equal proportion of classes in train/test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model(model_name)
    print(f"Training model: {model_name}")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\nModel Performance:")
    print(report_str)
    
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    metrics = {
        "model": model_name,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": float(acc),
        "per_class": {
            "0": {
                "precision": float(report_dict["0"]["precision"]),
                "recall": float(report_dict["0"]["recall"]),
                "f1": float(report_dict["0"]["f1-score"]),
                "support": int(report_dict["0"]["support"]),
            },
            "1": {
                "precision": float(report_dict["1"]["precision"]),
                "recall": float(report_dict["1"]["recall"]),
                "f1": float(report_dict["1"]["f1-score"]),
                "support": int(report_dict["1"]["support"]),
            },
        },
        "macro_avg": {
            "precision": float(report_dict["macro avg"]["precision"]),
            "recall": float(report_dict["macro avg"]["recall"]),
            "f1": float(report_dict["macro avg"]["f1-score"]),
            "support": int(report_dict["macro avg"]["support"]),
        },
        "weighted_avg": {
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1": float(report_dict["weighted avg"]["f1-score"]),
            "support": int(report_dict["weighted avg"]["support"]),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    return model, metrics

def save_model(model, filepath=None):
    if filepath is None:
        # Default to saving in the same directory as this script
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
        
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def save_metrics(metrics: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phishing URL Detector")
    parser.add_argument('dataset', help="Path to the training CSV")
    parser.add_argument('--model', default="rf", help="Model to train: rf | logreg | rf_calibrated | char_cnn")
    parser.add_argument('--out-dir', default="ml/runs", help="Directory where model + metrics are saved")
    parser.add_argument('--out', help="Path to save the model", default=None)
    parser.add_argument('--metrics-out', default=None, help="Path to save metrics JSON")

    args = parser.parse_args()
    
    df = load_data(args.dataset)
    
    def get_hostname(u):
        try:
            u = str(u)
            if not u.startswith(("http://", "https://")):
                u = "http://" + u
            return urlsplit(u).netloc.lower()
        except ValueError:
            return ""

    df["hostname"] = df["url"].apply(get_hostname)
    
    # Remove rows where hostname could not be parsed (worst results but needed)
    df = df[df["hostname"] != ""].reset_index(drop=True)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y=df["label"], groups=df["hostname"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    if args.model == "char_cnn":
        if CharCNN is None:
            raise ImportError("CharCNN requires tensorflow. Please install it.")
        
        print("Training Char-CNN (Deep Learning)...")
        # Standardize URLs for CNN (no feature extraction needed)
        urls_train = df_train["url"].astype(str).tolist()
        y_train = df_train["label"].astype(int).tolist()
        urls_test = df_test["url"].astype(str).tolist()
        y_test = df_test["label"].astype(int).tolist()

        cnn = CharCNN()
        cnn.build_model()
        cnn.train(urls_train, y_train, epochs=3, batch_size=64)
        
        # Evaluate
        print("\nEvaluating Char-CNN...")
        y_pred_prob = cnn.predict(urls_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        run_dir = os.path.join(args.out_dir, "char_cnn")
        os.makedirs(run_dir, exist_ok=True)
        cnn.save(run_dir)
        print(f"Char-CNN saved to {run_dir}")
        exit(0)

    X_train, y_train = extract_features_from_df(df_train, cache_path="ml/data/features_train_cache.csv")
    X_test,  y_test  = extract_features_from_df(df_test,  cache_path="ml/data/features_test_cache.csv")

    print(f"Train rows: {len(df_train)} | Test rows: {len(df_test)}")
    print(f"Unique hostnames train: {df_train['hostname'].nunique()} | test: {df_test['hostname'].nunique()}")

    model, metrics = train_model(X_train, y_train, X_test, y_test, model_name=args.model)
    run_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.pkl")
    metrics_path = os.path.join(run_dir, "metrics.json")

    save_model(model, model_path)
    save_metrics(metrics, metrics_path)

