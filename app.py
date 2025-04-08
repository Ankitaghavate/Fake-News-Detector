from flask import Flask, render_template, jsonify, request
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords

# 👉 Set nltk data path (for production deployment like Render)
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# ✅ Safe download of required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

# Initialize Flask app
app = Flask(__name__)

# ---------- Text Preprocessing Function ----------
def preprocessing(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    return " ".join(filtered_words)

# ---------- Load Model and Vectorizer ----------
model_path = os.path.join("model", "fake_news_detection.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"❌ Vectorizer not found at: {vectorizer_path}")

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

    try:
        cleaned_text = preprocessing(text)
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        result = "Real News" if prediction == 1 else "Fake News"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": f"Error analyzing text: {str(e)}"}), 500

# ---------- Run App ----------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
