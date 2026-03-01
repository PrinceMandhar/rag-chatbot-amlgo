import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------- LOAD DOCUMENT --------
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# -------- CHUNKING --------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

print(f"Total Chunks Created: {len(chunks)}")

# -------- LOAD EMBEDDING MODEL --------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- CREATE EMBEDDINGS --------
embeddings = embed_model.encode(
    chunks,
    normalize_embeddings=True
)

# -------- CREATE FAISS INDEX --------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

# -------- SAVE VECTOR DB --------
os.makedirs("vectordb", exist_ok=True)

faiss.write_index(index, "vectordb/faiss.index")

with open("vectordb/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector DB built successfully!")
