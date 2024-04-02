# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset

# %%
data = pd.read_csv("fake_or_real_news.csv")
data 

# %%
data["fake"] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)

# %%
X, y = data["title"], data["fake"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
# X_train_vectorized = vectorizer.fit_transform(X_train)
# X_test_vectorized = vectorizer.transform(X_test)

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


# %%
# clf = LinearSVC()
# clf.fit(X_train_vectorized, y_train)

# knn = KNeighborsClassifier()
# knn.fit(X_train_vectorized, y_train)


# Use the tokenizer in the same way as before
def encode_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Assuming data["title"] and y are prepared
X_encoded = encode_texts(data["title"].tolist())

# Prepare the dataset and DataLoader
dataset = TensorDataset(X_encoded['input_ids'], X_encoded['attention_mask'], torch.tensor(y, dtype=torch.long))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in loader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        model.zero_grad()
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

# %%
model.save_pretrained('./my_distilbert_model')
tokenizer.save_pretrained('./my_distilbert_model')

# %%

with open("mytext.txt", "r", encoding="utf-8") as f:
    text = f.read()





# %%



