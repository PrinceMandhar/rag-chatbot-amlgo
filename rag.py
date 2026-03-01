import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAGChatbot:

    def __init__(self):

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loading vector DB...")
        self.index = faiss.read_index("vectordb/faiss.index")

        print("Loading chunks...")
        with open("vectordb/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loading LLM...")
        self.model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def retrieve(self, query, top_k=4):

        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        )

        distances, indices = self.index.search(
            np.array(query_embedding),
            top_k
        )

        retrieved_chunks = []
        for i in indices[0]:
            if i < len(self.chunks):
                retrieved_chunks.append(self.chunks[i])

        return retrieved_chunks

    def get_response(self, query):

        retrieved_chunks = self.retrieve(query)

        if not retrieved_chunks:
            return "I don't know based on the document."

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
Answer the question strictly using the provided context.
Do not guess.
If the answer is not clearly present, say:
"I don't know based on the document."

Context:
{context}

Question:
{query}
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2
        )

        result = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return result.strip()
