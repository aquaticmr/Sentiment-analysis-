import flask
from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load the Saved Model ---
# Ensure the model file is in the same directory as this script
model_filename = 'sentiment_pipeline.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# --- Download NLTK data required for cleaning ---
# This is a good practice for deployment environments
# The corrected line
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab'], quiet=True)

# --- Define the SAME Text Cleaning Function from Training ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_for_nb(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(words)

# --- Define Web Routes ---

# Main page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get text from the form
        review_text = request.form['review']

        # Clean the text using the same function
        cleaned_review = clean_text_for_nb(review_text)

        # Make prediction using the loaded pipeline
        # The pipeline expects an iterable (like a list), so we pass [cleaned_review]
        prediction_code = model.predict([cleaned_review])[0]
        prediction = "Positive" if prediction_code == 1 else "Negative"

        # Render the page again with the prediction result
        return render_template('index.html', prediction=prediction, review=review_text)

# --- Run the App ---
if __name__ == '__main__':
    # You might need to install these libraries:
    # pip install Flask scikit-learn pandas nltk
    app.run(debug=True)