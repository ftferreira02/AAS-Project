import sys
import os
import pickle
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml'))
from features import FeatureExtractor

# Imports for CharCNN
try:
    import tensorflow as tf
    from char_cnn import CharCNN
except ImportError:
    print("TensorFlow not installed.")
    sys.exit(1)

MODEL_PATH = "ml/runs/xgb_calibrated/model.pkl"
CNN_PATH = "ml/runs/char_cnn"

def main():
    url = "https://github.com/detiuaveiro/aas/blob/main/slides/slides_04.pdf"
    
    print(f"Analyzing URL: {url}")
    
    # 1. Lexical
    print("\n--- Lexical Analysis ---")
    extractor = FeatureExtractor(url)
    features = extractor.get_features()
    print("Features extracted:")
    for k, v in features.items():
        print(f"  {k}: {v}")
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    df = pd.DataFrame([features])
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)
        
    lex_prob = model.predict_proba(df)[0][1]
    print(f"Lexical Probability: {lex_prob:.4f}")

    # 2. CNN
    print("\n--- CNN Analysis ---")
    cnn = CharCNN()
    cnn.load(CNN_PATH)
    cnn_prob = float(cnn.predict(url)[0])
    print(f"CNN Probability: {cnn_prob:.4f}")

    # 3. Ensemble
    final_score = (0.4 * lex_prob) + (0.6 * cnn_prob)
    print(f"\n--- Final Ensemble Score ---")
    print(f"0.4 * {lex_prob:.4f} + 0.6 * {cnn_prob:.4f} = {final_score:.4f}")

if __name__ == "__main__":
    main()
