from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [float(data[f]) for f in feature_names]
        prob = model.predict_proba([features])[0]
        ai_prob = prob[1]
        label = 'AI-Generated' if ai_prob > 0.5 else 'Human-Authored'
        return jsonify({
            'label': label,
            'ai_probability': round(ai_prob * 100, 1),
            'human_probability': round(prob[0] * 100, 1),
            'confidence': round(max(prob) * 100, 1),
            'top_feature': 'CN/T',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
