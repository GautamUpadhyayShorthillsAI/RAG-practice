import os
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from dotenv import load_dotenv  # ✅ Import dotenv

# Load environment variables
load_dotenv()

# Step 1: Set up Gemini API key
api_key = os.getenv("GEMINI_API_KEY")  # Load API key from environment variable
if not api_key:
    raise ValueError("Missing Gemini API key. Set it as an environment variable.")

# Configure Gemini API
genai.configure(api_key=api_key)

FAISS_INDEX_PATH = "faiss_index.faiss"
METADATA_PATH = "metadata.pkl"

# Step 2: Read all chunked text files from a directory
def read_chunks_from_folder(folder_path: str):
    print('✅ Reading chunked text files...')
    chunks, filenames = [], []
    
    for file in sorted(os.listdir(folder_path)):  # Sort to keep order
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                chunk = f.read().strip()
                if chunk:
                    chunks.append(chunk)
                    filenames.append(file)  # Store file name as metadata
    
    return chunks, filenames

# Step 3: Generate embeddings using Gemini API
def generate_embedding(text: str) -> np.ndarray:
    print('🧠 Generating embedding using Gemini...')
    
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use text generation model
    
    # Request Gemini to generate an embedding-like response
    response = model.generate_content(f"Generate a dense vector representation for: {text}")
    
    # Convert the response text into a numerical vector (dummy embedding)
    embedding = np.array([ord(char) for char in response.text[:1536]], dtype=np.float32)

    return embedding



# Step 4: Initialize FAISS index
def initialize_faiss_index(dimension: int) -> faiss.Index:
    print('🗂️ Initializing FAISS index...')
    return faiss.IndexFlatL2(dimension)

# Step 5: Save FAISS index
def save_faiss_index(index: faiss.Index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print("💾 FAISS index saved!")

# Step 6: Load FAISS index
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print("🔄 FAISS index loaded!")
        return index, metadata
    return None, None

# Step 7: Add embeddings to FAISS
def add_to_faiss(index: faiss.Index, embeddings: list):
    print('📥 Adding embeddings to FAISS...')
    if embeddings:
        index.add(np.array(embeddings, dtype=np.float32))

# Step 8: Perform similarity search
def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int = 5):
    print('🔍 Searching FAISS index...')
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
    return indices[0]  # Return the indices of the top-k most similar documents

# Step 9: Query Gemini LLM with Context
def query_llm(query: str, context: list):
    print('🤖 Querying Gemini AI...')
    context_str = "\n\n".join(context)  # Combine retrieved chunks into one prompt
    prompt = f"""
    You are a knowledgeable assistant. Given the following context, answer the user's question accurately.

    Context:
    {context_str}

    Question: {query}

    Answer:
    """

    model = genai.GenerativeModel("gemini-1.5-pro")  # ✅ Corrected model name
    response = model.generate_content(prompt)

    return response.text.strip()

# Step 10: Main Function
def main():
    folder_path = "data"
    faiss_index, metadata_list = load_faiss_index()
    
    if faiss_index is None:
        print("🚀 Creating a new FAISS index...")
        chunks, filenames = read_chunks_from_folder(folder_path)
        if not chunks:
            print("❌ No valid text chunks found in the folder.")
            return
        
        dimension = 1536
        faiss_index = initialize_faiss_index(dimension)
        metadata_list = []
        embeddings = [generate_embedding(chunk) for chunk in chunks]
        add_to_faiss(faiss_index, embeddings)
        metadata_list = [{"file": filenames[i], "text": chunks[i]} for i in range(len(chunks))]
        save_faiss_index(faiss_index, metadata_list)

    return faiss_index, metadata_list

if __name__ == "__main__":
    main()
