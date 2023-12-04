import os
import glob
import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

data_folder = "/content/network-security-white-labelling/basic_nlp/data"


data = []
labels = []

for label_folder in os.listdir(data_folder):
    label_path = os.path.join(data_folder, label_folder)

    if os.path.isdir(label_path):
        data_files = glob.glob(os.path.join(label_path, "data.txt"))

        if data_files:
            with open(data_files[0], "r", encoding="utf-8") as file:
                text = file.read()
                data.append(text)
                labels.append(label_folder)


tokenized_data = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in data]
embeddings = []

for item in tokenized_data:
    with torch.no_grad():
        output = model(**item)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())

num_clusters = 3  
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(embeddings)


clustered_data = {}
for i, label in enumerate(labels):
    cluster = cluster_labels[i]
    if cluster not in clustered_data:
        clustered_data[cluster] = []
    clustered_data[cluster].append(label)


print(clustered_data)
