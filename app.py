
import streamlit as st
from rag import RAGChatbot

st.set_page_config(page_title="Document RAG Chatbot", layout="wide")
st.title("📚 Document Based RAG Chatbot")

@st.cache_resource
def load_bot():
    return RAGChatbot()

bot = load_bot()

questions = [
    "What is the purpose of this document?",
    "What are the key objectives?",
    "What are the payment terms?",
    "What are the termination conditions?"
]

st.subheader("💡 Try These Questions")

for q in questions:
    if st.button(q):
        st.session_state.user_input = q

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.user_input = user_input

if "user_input" in st.session_state:

    query = st.session_state.user_input

    st.markdown(f"### 🧑 You: {query}")

    with st.spinner("Thinking..."):
        response = bot.get_response(query)

    st.markdown(f"### 🤖 Bot: {response}")
