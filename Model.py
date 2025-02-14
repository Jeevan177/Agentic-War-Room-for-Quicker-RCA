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

class IssuesDataset(Dataset):
    def __init__(self, issues, labels, tokenizer):
        self.issues = issues
        self.labels = labels
        self.tokenizer = tokenizer
        # Initialize LabelEncoder to convert labels to numerical representation
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels) # Fit and transform labels

    def __len__(self):
        return len(self.issues)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.issues[idx], truncation=True, padding='max_length', max_length=512)
        item = {key: torch.tensor(val) for key, val in encoding.items()}

        item['labels'] = torch.tensor(self.labels[idx]) # Use encoded labels
        return item

# Custom collate function for padding
def collate_fn(batch):
    # Use the tokenizer's pad method to pad the batch
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # Return the padded batch
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Sample dataset (Make sure 'labels' matches the number of issues)
df = pd.read_csv("issues_dataset.csv")
issues = df['Issue'].tolist()  # The issues
labels = df['Root Causes'].tolist()  # Replace with the correct column for labels

# Ensure that issues and labels have the same length
assert len(issues) == len(labels), f"Length mismatch: {len(issues)} issues vs {len(labels)} labels."

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# Get the number of unique labels after encoding
num_labels = len(set(labels))
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Create dataset and dataloader
dataset = IssuesDataset(issues, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop (basic example)
for epoch in range(10):  # Adjust epochs as needed
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed with loss: {loss.item()}")