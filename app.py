import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model
with open('model/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    return jsonify({
        'text': text,
        'label': int(prediction),
        'sentiment': 'positive' if prediction == 1 else 'negative',
        'confidence': round(confidence * 100, 1)
    })

if __name__ == '__main__':
    app.run(debug=True)