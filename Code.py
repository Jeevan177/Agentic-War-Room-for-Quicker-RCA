import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import time

# Load the issue dataset and model
df = pd.read_csv('issues_dataset.csv')
df = df.replace({'\\[': '', '\\]': '', "''": ''}, regex=True).infer_objects(copy=False).astype(str)
df = df.iloc[0:20]
model = DistilBertForSequenceClassification.from_pretrained("model")
tokenizer = DistilBertTokenizer.from_pretrained("tokenizer")

# Streamlit UI
st.title("Agentic War Room for Quicker RCA")

text = """The ***Agentic War Room for Quicker RCA*** leverages machine learning models, such as **DistilBERT** which is a basic language model with less parameters,
 to classify issues, predict root causes, and generate solutions in real time."""

def stream_data():
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# if st.button("Stream data"):
st.write_stream(stream_data)

# Select issue from dropdown
selected_issue = st.selectbox("Select an Issue:", df['Issue'].unique())

# Tokenize the selected issue
inputs = tokenizer(selected_issue, return_tensors="pt", truncation=True, padding=True)

# Perform inference with DistilBERT
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

# Fetch predicted root cause & solution
predicted_root_cause = df['Root Causes'].iloc[predicted_label]
predicted_solution = df['Solutions'].iloc[predicted_label]

# Format root cause and solution
if isinstance(predicted_root_cause, (list, tuple)):
    predicted_root_cause = ", ".join(predicted_root_cause)

if isinstance(predicted_solution, (list, tuple)):
    predicted_solution = ", ".join(predicted_solution)

# Display results
st.subheader("Predicted Root Cause(s):")
st.write(predicted_root_cause)

st.subheader("Suggested Solution(s):")
st.write(predicted_solution)
