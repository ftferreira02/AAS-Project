from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import sys
import os
import pandas as pd

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "ml", "runs", "rf_calibrated", "model.pkl")
)
model = None

def get_expected_feature_names(m):
    # Plain estimators trained with pandas
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)

    # Pipeline (e.g., scaler + logistic regression)
    if hasattr(m, "named_steps"):
        # try the last step (classifier)
        last = list(m.named_steps.values())[-1]
        if hasattr(last, "feature_names_in_"):
            return list(last.feature_names_in_)

    # CalibratedClassifierCV (new sklearn uses .estimator)
    if hasattr(m, "estimator") and hasattr(m.estimator, "feature_names_in_"):
        return list(m.estimator.feature_names_in_)

    # Fallback for some calibrated objects
    if hasattr(m, "calibrated_classifiers_") and m.calibrated_classifiers_:
        base = m.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_names_in_"):
            return list(base.feature_names_in_)

    return None

def load_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

# Add ml folder to path to import features.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
from features import FeatureExtractor

app = Flask(__name__)
# Security: Only allow requests from extensions (or localhost for dev)
CORS(app, resources={r"/predict": {"origins": ["chrome-extension://*", "http://localhost:*", "http://127.0.0.1:*"]}})

# Load model on startup
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.get_json()
    if not data:
         return jsonify({'error': 'Invalid JSON'}), 400

    url = data.get('url', '')
    
    # Input Validation
    if not url or not isinstance(url, str):
        return jsonify({'error': 'Invalid URL format'}), 400
    if len(url) > 2000:
        return jsonify({'error': 'URL too long'}), 400
        
    # Normalization
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
        
    try:
        # Extract features
        extractor = FeatureExtractor(url)
        features = extractor.get_features()
        
        # Convert to DataFrame for model
        input_df = pd.DataFrame([features])
        
        # FIX: Ensure columns are in the exact same order as training
        # This prevents silent errors where "url_length" might overlap with "entropy"
        if hasattr(model, "feature_names_in_"):
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        else:
            # Fallback (unlikely needed with recent sklearn)
            input_df = input_df.reindex(sorted(input_df.columns), axis=1, fill_value=0)
        
        # Predict Probabilities
        # Class 0 = Safe, Class 1 = Phishing
        proba = model.predict_proba(input_df)[0]
        phishing_prob = float(proba[1])
        
        # Decision Policy (3-Level)
        # Safe:    prob < 0.60
        # Warning: 0.60 <= prob < 0.85
        # Unsafe:  prob >= 0.85
        
        if phishing_prob < 0.60:
            level = "safe"
            is_phishing = False
        elif phishing_prob < 0.85:
            level = "warning"
            is_phishing = True # Trigger the extension, but we'll show a warning UI
        else:
            level = "unsafe"
            is_phishing = True

        result = {
            'url': url,
            'is_phishing': is_phishing,
            'phishing_probability': phishing_prob,
            'confidence': max(proba),
            'level': level, # New field
            'features': features
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    load_model()
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Bind to localhost explicitly for security
    app.run(host='127.0.0.1', port=5000, debug=True)
