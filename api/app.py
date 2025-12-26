from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import sys
import os
import pandas as pd

# Add ml folder to path to import features.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
from features import FeatureExtractor

app = Flask(__name__)
CORS(app)  # Enable CORS for extension

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
        
    try:
        # Extract features
        extractor = FeatureExtractor(url)
        features = extractor.get_features()
        
        # Convert to DataFrame for model
        input_df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(input_df)[0]
        # Get probabilities if available
        try:
            proba = model.predict_proba(input_df)[0]
            confidence = float(max(proba))
        except:
            confidence = 1.0
            
        result = {
            'url': url,
            'is_phishing': bool(prediction == 1),
            'confidence': confidence,
            'features': features
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
