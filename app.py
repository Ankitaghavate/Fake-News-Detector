from flask import Flask, render_template, jsonify, request
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define Flask app
app = Flask(__name__)

# ---------- Preprocessing Function ----------
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# ---------- Load Model and Vectorizer ----------
model_path = os.path.join(os.getcwd(), "model", "fake_news_detection.pkl")
vectorizer_path = os.path.join(os.getcwd(), "model", "tfidf_vectorizer.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template("index.html")  # Make sure templates/index.html exists

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Apply preprocessing
    cleaned_text = preprocessing(text)

    # Transform and predict
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": result})

# ---------- Run App ----------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
