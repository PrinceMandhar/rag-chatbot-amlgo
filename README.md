# RAG Chatbot with Streaming Response  
## Junior AI Engineer Assignment – Amlgo Labs

---

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user queries using a provided document.

The system converts the document into semantic embeddings, stores them in a FAISS vector database, retrieves relevant content using similarity search, and generates grounded answers using an instruction-tuned open-source LLM.

The chatbot supports real-time streaming responses via Streamlit.

---

## 2. Folder Structure

/data → Original document
/chunks → Processed text chunks
/vectordb → Saved FAISS index
/src → Retriever and generator logic
app.py → Streamlit chatbot application
build_index.py → Script to create embeddings and FAISS index
requirements.txt
README.md

yaml
Copy code

---

## 3. How the System Works

1. Document is cleaned and split into 100–300 word chunks  
2. Each chunk is converted into embeddings using SentenceTransformers  
3. Embeddings are stored in FAISS vector database  
4. User query is converted into embedding  
5. Similar chunks are retrieved using similarity search  
6. Retrieved chunks are injected into prompt  
7. LLM generates grounded response  
8. Response is streamed in real time  

---

## 4. Technologies Used

- Embedding Model: all-MiniLM-L6-v2  
- Vector Database: FAISS  
- LLM: Instruction-tuned open-source model  
- Framework: Streamlit  
- Language: Python  

---

## 5. How to Run the Project

### Step 1 – Install Dependencies

pip install -r requirements.txt

shell
Copy code

### Step 2 – Build Vector Index

python build_index.py

shell
Copy code

This step creates embeddings and saves FAISS index.

### Step 3 – Run Chatbot

streamlit run app.py

yaml
Copy code

Open in browser:
http://localhost:8501

---

## 6. Key Features

- Natural language query input  
- Real-time streaming response  
- Retrieval-based grounded answers  
- Display of source chunks  
- Reset chat functionality  

---

## 7. Key Learnings

- Document preprocessing and chunking  
- Semantic embedding generation  
- FAISS similarity search  
- RAG pipeline implementation  
- Prompt engineering  
- Streaming response handling  

---

## 8. Future Improvements

- Hybrid search (keyword + embedding)  
- Larger instruction-tuned model  
- Cloud deployment  
- Response confidence scoring  