import os
import faiss
import openai
import numpy as np
import pickle  # For saving metadata
from dotenv import load_dotenv

from openai import AzureOpenAI

# Load environment variables
load_dotenv()

<<<<<<< HEAD
class FAISSRAG:
    def __init__(self, index_path="faiss_index.faiss", metadata_path="metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key. Set it as an environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.index, self.metadata = self.load_faiss_index()
    
    def read_chunks_from_folder(self, folder_path: str):
        print('✅ Reading chunked text files...')
        chunks, filenames = [], []
=======
# Step 1: Set up OpenAI API key
# api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment variable
# if not api_key:
#     raise ValueError("Missing OpenAI API key. Set it as an environment variable.")

# Initialize OpenAI client
# client = openai.OpenAI(api_key=api_key)
client = openai.AzureOpenAI(api_key="", azure_endpoint="", api_version="", default_headers={"User-ID": ""})

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

# Step 3: Generate embeddings
def generate_embedding(text: str) -> np.ndarray:
    print('🧠 Generating embedding...')
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding)

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

# Step 9: Query LLM with Context
def query_llm(query: str, context: list):
    print('🤖 Querying LLM...')
    context_str = "\n\n".join(context)  # Combine relevant chunks into one string
    prompt = f"""
    You are a knowledgeable assistant. Given the following context, answer the user's question accurately.

    Context:
    {context_str}

    Question: {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

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
>>>>>>> 55de148 (Updated the main.py)
        
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk = f.read().strip()
                    if chunk:
                        chunks.append(chunk)
                        filenames.append(file)  # Store file name as metadata
        
        return chunks, filenames

    def generate_embedding(self, text: str) -> np.ndarray:
        print('🧠 Generating embedding...')
        response = self.client.embeddings.create(
            model="text-embedding-3-small",  # Updated model
            input=text
        )
        return np.array(response.data[0].embedding)

    def initialize_faiss_index(self, dimension: int):
        print('🗂️ Initializing FAISS index...')
        return faiss.IndexFlatL2(dimension)
    
    def save_faiss_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print("💾 FAISS index saved!")
    
    def load_faiss_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
            print("🔄 FAISS index loaded!")
            return index, metadata
        return None, None
    
    def add_to_faiss(self, embeddings: list):
        print('📥 Adding embeddings to FAISS...')
        if embeddings:
            self.index.add(np.array(embeddings, dtype=np.float32))
    
    def search_faiss(self, query_embedding: np.ndarray, k: int = 5):
        print('🔍 Searching FAISS index...')
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return indices[0]
    
    def query_llm(self, query: str, context: list):
        print('🤖 Querying LLM...')
        context_str = "\n\n".join(context)
        prompt = f"""
        You are a knowledgeable assistant. Given the following context, answer the user's question accurately.
        
        Context:
        {context_str}
        
        Question: {query}
        
        Answer:
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",  # Updated model
            messages=[{"role": "system", "content": "You are an AI assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    
    def build_index(self, folder_path: str):
        if self.index is None:
            print("🚀 Creating a new FAISS index...")
            chunks, filenames = self.read_chunks_from_folder(folder_path)
            if not chunks:
                print("❌ No valid text chunks found in the folder.")
                return
            
            dimension = 1536
            self.index = self.initialize_faiss_index(dimension)
            self.metadata = []
            embeddings = [self.generate_embedding(chunk) for chunk in chunks]
            self.add_to_faiss(embeddings)
            self.metadata = [{"file": filenames[i], "text": chunks[i]} for i in range(len(chunks))]
            self.save_faiss_index()
        return self.index, self.metadata

if __name__ == "__main__":
    rag = FAISSRAG()
    rag.build_index("data")
