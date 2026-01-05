import sys
import os
import pickle
import pandas as pd
import numpy as np

# Add ml folder to path
sys.path.append(os.path.join(os.getcwd(), 'ml'))
from features import FeatureExtractor
from char_cnn import CharCNN

MODEL_PATH = "ml/runs/xgb_calibrated/model.pkl"
CNN_PATH = "ml/runs/char_cnn"

def test_url(url):
    print(f"\n--- Testing {url} ---")
    
    # 1. Feature Extraction
    features = None
    try:
        extractor = FeatureExtractor(url)
        features = extractor.get_features()
        print("Features extracted successfully.")
        # Print a few key features to verify they look sane
        print(f"Features (Head): length={features.get('url_length')}, dots={features.get('count_dots')}, is_https={features.get('is_https')}")
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    # 2. Lexical Model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
            
        print(f"Model Type: {type(model)}")
        
        # Check classes if available
        if hasattr(model, "classes_"):
            print(f"Model Classes: {model.classes_}")
            
        # Unwrap CalibratedClassifierCV to check base estimator if needed
        base_model = model.estimator if hasattr(model, "estimator") else model
        print(f"Base Model: {type(base_model)}")
        
        input_df = pd.DataFrame([features])
        
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        else:
            input_df = input_df.reindex(sorted(input_df.columns), axis=1, fill_value=0)
            
        prob = model.predict_proba(input_df)[0]
        print(f"Lexical Probabilities: {prob}")
        print(f"Lexical Phishing Prob (Class 1?): {prob[1]}")
    except Exception as e:
        print(f"Lexical model failed: {e}")

    # 3. CNN Model
    try:
        cnn = CharCNN()
        cnn.load(CNN_PATH)
        prob_cnn = cnn.predict([url])[0]
        print(f"CNN Probability: {prob_cnn}")
    except Exception as e:
        print(f"CNN model failed: {e}")

if __name__ == "__main__":
    test_url("http://google.com")
    test_url("https://www.google.com") # dataset sample style
    test_url("http://g0ogle.com")
