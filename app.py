from flask import Flask, render_template, jsonify, request
import joblib  # Use joblib for saving/loading
import os

# Construct absolute paths (make sure these are correct for your file structure)
model_path = os.path.join(os.getcwd(), "model", "fake_news_detection.pkl") 
vectorizer_path = os.path.join(os.getcwd(), "model", "tfidf_vectorizer.pkl") 


# Load the model using joblib
with open(model_path, "rb") as f:
    model = joblib.load(f)

# Load the TF-IDF vectorizer using joblib
with open(vectorizer_path, "rb") as f:
    tfidf_vectorizer = joblib.load(f)

# Initialize Flask app
app = Flask(__name__)

# Homepage route
@app.route('/')
def home():
    return render_template("index.html")  # Assumes 'index.html' is in 'templates' folder

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text") 

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform the input using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Format result
    result = "Fake News" if prediction == 1 else "Real News"

    return jsonify({"prediction": result})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
