# 📰 Fake News Detector 🔍

This is a web-based Fake News Detection system built using **Natural Language Processing (NLP)** and an **Artificial Neural Network (ANN)**. It classifies news articles as **real** or **fake** based on their content.

---

## 🚀 Features

- Text preprocessing using NLP techniques
- ANN model for binary classification
- Simple web interface using Flask
- Easy deployment (Render/GitHub)

---

## 🧠 Tech Stack

- **Python**
- **Flask** – Web Framework
- **ANN (Artificial Neural Network)** – For classification
- **NLP** – For text cleaning
- **Scikit-learn**, **TensorFlow/Keras**, **Pandas**, **NumPy**
- **HTML, CSS** – Frontend
- **Render** – Deployment

---

## 🛠️ How to Run This Project Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Fake-News-Detector.git
cd Fake-News-Detector
```


### 2. Create a Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Run the Application
```bash
python app.py
Visit: http://127.0.0.1:5000/ in your browser.
```

### 5.🧹 Text Preprocessing (NLP)
Removing punctuation and stopwords

-Lowercasing

-Tokenization

-Lemmatization

### 6. 🧠 Machine Learning Model
Model Type: Artificial Neural Network

-Framework: Keras

-Trained on labeled fake/real news dataset

-Accuracy: 91.02%

- Fake News: Trump supporters and the so-called president s favorite network are lashing out at special counsel Robert Mueller and the FBI.
- Real News:- BRUSSELS (Reuters) - NATO allies on Tuesday welcomed President Donald Trump s decision to commit more forces to Afghanistan, as part of a new U.S. strategy he said would require more troops and funding from America s partners. 

### 7. 📁 Folder Structure
```bash
├── model/             # Saved trained model
├── templates/         # HTML frontend
├── app.py             # Flask backend
├── render.yaml        # Deployment config
├── requirements.txt   # Python dependencies
└── .gitignore

```

Feel free to connect or contribute!
