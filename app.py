import os
import pickle
import string
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# -------------------------
# Download NLTK resources safely
# -------------------------
nltk.download('punkt')
nltk.download('punkt_tab')   # Needed for Python 3.12+
nltk.download('stopwords')

# -------------------------
# Initialize Flask
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Model & Vectorizer
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------------
# Text Preprocessing Function
# -------------------------
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    # Remove special characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords + punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        message = request.form.get("message")

        if message:
            transformed = transform_text(message)
            vector_input = vectorizer.transform([transformed])
            result = model.predict(vector_input)[0]

            if result == 1:
                prediction = "🚨 SPAM MESSAGE"
            else:
                prediction = "✅ NOT SPAM"

    return render_template("index.html", prediction=prediction)


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)