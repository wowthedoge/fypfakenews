import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pandas as pd

df = pd.read_csv("datasets/fakenewsnet.csv")

df.fillna({'text': 'None', 'label': 'None'}, inplace=True)  # Impute missing values appropriately

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)  # Added random_state for reproducibility
val_df = val_df.reset_index(drop=True)

# Vectorize the text using TF-IDF
tfidf_vc = TfidfVectorizer(min_df=10, max_features=100000, analyzer="word", ngram_range=(1, 2), stop_words='english', lowercase=True)
train_vc = tfidf_vc.fit_transform(train_df["text"])
val_vc = tfidf_vc.transform(val_df["text"])

# Train a logistic regression model
model = LogisticRegression(C=0.5, solver="sag")
model.fit(train_vc, train_df['label'])  

# Predict and evaluate the model
val_pred = model.predict(val_vc)
val_score = f1_score(val_df['label'], val_pred, average="binary")  
print(val_score)

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vc, 'tfidf_vectorizer.pkl')

# Save the trained model
joblib.dump(model, 'lime_ml_model.pkl')