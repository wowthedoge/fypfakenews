from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
# You can load these objects from disk or define them here if they're small

# Load the trained model
model = joblib.load('lime_ml_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vc = joblib.load('tfidf_vectorizer.pkl')

pipeline = make_pipeline(tfidf_vc, model)

# Set up LIME explainer
class_names = ["Real", "Fake"]
explainer = LimeTextExplainer(class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input text from the frontend
    input_text = request.json['text']

    # Perform prediction
    prediction_proba = pipeline.predict_proba([input_text])[0, 1]

    # Generate explanation using LIME
    exp = explainer.explain_instance(input_text, pipeline.predict_proba, num_features=15)

    # Format the explanation for display
    explanation = exp.as_list()

    print("SAVING TO HTML")
    exp.save_to_file('temp.html')

    # Prepare response
    response = {
        "prediction_fake": prediction_proba.tolist(),
        "explanation": explanation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
