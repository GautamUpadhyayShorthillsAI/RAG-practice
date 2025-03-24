import streamlit as st
import logging
from main import FAISSRAG  # Import the FAISSRAG class

# Setup logging for queries
query_log_file = "query.log"
logging.basicConfig(
    filename=query_log_file,
    level=logging.INFO,
    format="%(asctime)s - QUERY: %(message)s",
    filemode="w"
)
query_logger = logging.getLogger("query_logger")
query_logger.propagate = False  # Prevent duplicate logs

# Initialize FAISS RAG
rag = FAISSRAG()
faiss_index, metadata_list = rag.load_faiss_index()

if faiss_index is None:
    st.warning("⚠️ No FAISS index found. Initializing and processing documents...")
    faiss_index, metadata_list = rag.build_index("data")

# Streamlit UI setup
st.set_page_config(page_title="🔍 NCERT Science", layout="wide")
st.title("🔍 NCERT Science")
st.markdown("**Enter a query to search through indexed chunks and get a response.**")

# Query Input
user_query = st.text_input("📝 Enter your query", "")

if st.button("🔎 Search"):
    if not user_query.strip():
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("🔍 Searching..."):
            # Generate query embedding
            query_embedding = rag.generate_embedding(user_query)
            similar_indices = rag.search_faiss(query_embedding, k=5)

            # Retrieve relevant chunks
            retrieved_chunks = [metadata_list[idx]["text"] for idx in similar_indices if 0 <= idx < len(metadata_list)]

            # Get answer from LLM
            final_answer = rag.query_llm(user_query, retrieved_chunks)

        # Display the final AI-generated answer
        st.subheader("💡 AI-Generated Answer")
        st.write(final_answer)

        # Log the user query and AI response
        query_logger.info(f"User Query: {user_query} | AI Response: {final_answer}")
