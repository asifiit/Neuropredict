from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load your trained model here — update the path as needed
model_path = os.path.join('models', 'best_xgboost_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    # You can keep this rendering prediction.html or change as you like
    return render_template('prediction.html')

@app.route('/predict-page')
def prediction_page():
    return render_template('prediction.html')

@app.route('/index-page')
def index_page():
    # This renders your index.html page (e.g., info, questionnaire, etc)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')

        if features is None:
            return jsonify({'error': 'No features provided.'})

        features = np.array(features).reshape(1, -1).astype(float)

        if features.shape[1] != model.n_features_in_:
            return jsonify({'error': f'Expected {model.n_features_in_} features, got {features.shape[1]}.'})

        prediction = model.predict(features)[0]

        result_message = "The patient has Alzheimer's." if prediction == 1 else "The patient does not have Alzheimer's."

        return jsonify({'prediction': result_message})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
