from flask import Flask, request, jsonify
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import requests
import os.path
import joblib
from tqdm import tqdm

def download_file(url, filename):

    if os.path.isfile(filename):  # Check if the file already exists
        print(f"File '{filename}' already exists. Skipping download.")
        return
    # Make a request to download the file
    response = requests.get(url, stream=True)
    
    # Get the total file size from the response headers
    total_size = int(response.headers.get('content-length', 0))
    
    # Initialize the progress bar
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    # Write the downloaded data to a file
    with open(filename, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            progress_bar.update(len(data))
    
    # Close the progress bar
    progress_bar.close()

try:
    model_url = 'https://github.com/wowthedoge/fypfakenews/releases/download/v1.0/lime_ml_model.pkl'
    model_filename = 'lime_ml_model.pkl'
    print("Downloading model...")
    download_file(model_url, model_filename)
    print("Model downloaded. Loading model...")
    model = joblib.load(model_filename)
    print("Model loaded.")
except Exception as e:
    print(e)

try:
    vectorizer_url = 'https://github.com/wowthedoge/fypfakenews/releases/download/v1.0/tfidf_vectorizer.pkl'
    vectorizer_filename = 'tfidf_vectorizer.pkl'
    print("Downloading vectorizer...")
    download_file(vectorizer_url, vectorizer_filename)
    print("Vectorizer downloaded. Loading vectorizer...")
    vectorizer = joblib.load(vectorizer_filename)
    print("Vectorizer loaded.")
except Exception as e:
    print(e)

app = Flask(__name__)

pipeline = make_pipeline(vectorizer, model)

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

    # Prepare response
    response = {
        "prediction_fake": prediction_proba.tolist(),
        "explanation": explanation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
