# main.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env_region = os.getenv("PINECONE_ENVIRONMENT")  # Now "us-east-1"

# Initialize Pinecone client using the new SDK
pc = Pinecone(api_key=api_key)

# Define your index name
index_name = "imdb-movie-vecs"

# Create the index if it doesn't exist, specifying AWS cloud with region "us-east-1"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,      # Dimension of embeddings from all-MiniLM-L6-v2
        metric="cosine",    # Using cosine similarity
        spec=ServerlessSpec(cloud="aws", region=env_region)
    )

# Connect to the index
index = pc.Index(index_name)

# Load and preprocess the dataset
df = pd.read_csv("imdb_top_1000.csv")
df.dropna(subset=["Series_Title", "Genre", "Overview"], inplace=True)
df["text"] = df["Series_Title"] + " " + df["Genre"] + " " + df["Overview"]

# Initialize SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Generating embeddings...")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# Prepare vectors for upsert: each vector has an ID, values, and metadata
vectors = []
for i in range(len(embeddings)):
    vectors.append({
        "id": str(i),
        "values": embeddings[i].tolist(),
        "metadata": {
            "title": df.iloc[i]["Series_Title"],
            "genre": df.iloc[i]["Genre"],
            "overview": df.iloc[i]["Overview"]
        }
    })

# Batch upload (in batches of 100) into the Pinecone index under the "default" namespace
batch_size = 100
print("Uploading embeddings to Pinecone...")
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(
        vectors=batch,
        namespace="default"
    )

print("✅ Embeddings uploaded to Pinecone!")
