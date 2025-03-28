import streamlit as st
import logging
from main import FAISSRAG  # Import the FAISSRAG class

# Setup logging for queries
query_log_file = "query.log"
logging.basicConfig(
    filename=query_log_file,
    level=logging.INFO,
    format="%(asctime)s - QUERY: %(message)s",
    filemode="a"
)
query_logger = logging.getLogger("query_logger")
query_logger.propagate = False  # Prevent duplicate logs

# Initialize FAISS RAG
rag = FAISSRAG()
faiss_index, metadata_list = rag.load_faiss_index()

if faiss_index is None:
    st.warning("⚠️ No FAISS index found. Initializing and processing documents...")
    rag.build_index("data")  # Rebuild index
    faiss_index, metadata_list = rag.load_faiss_index()

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

            # Retrieve relevant chunks (Fixing metadata retrieval)
            retrieved_chunks = [
                metadata_list[idx]["text"] if isinstance(metadata_list[idx], dict) else metadata_list[idx]
                for idx in similar_indices if 0 <= idx < len(metadata_list)
            ]

            # Display Retrieved Chunks in Collapsibles
            st.subheader("📚 Retrieved Chunks")
            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    with st.expander(f"🔎 Chunk {i}"):
                        st.write(chunk)
            else:
                st.warning("⚠️ No relevant chunks found. The AI may not be able to answer.")

            # Ensure the LLM only answers from retrieved context
            if retrieved_chunks:
                context_text = "\n\n".join(retrieved_chunks)
                final_answer = rag.query_llm(user_query, context_text)  # Pass question and context
            else:
                final_answer = "I don't have the context for it."

        # Display the final AI-generated answer
        st.subheader("💡 AI-Generated Answer")
        st.write(final_answer)

        # Log the user query and AI response
        query_logger.info(f"User Query: {user_query} | AI Response: {final_answer}")
