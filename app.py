from flask import Flask, render_template, jsonify, request
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords

# Safe download of NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# ---------- Text Preprocessing Function ----------
def preprocessing(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = nltk.word_tokenize(text)  # Tokenize

    stop_words = set(stopwords.words('english'))  # Load stopwords
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords

    return " ".join(filtered_words)

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
    return render_template("index.html")  # Ensure templates/index.html exists

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess input
    cleaned_text = preprocessing(text)

    # Transform & Predict
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)[0]
    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({"prediction": result})

# ---------- Run App ----------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
