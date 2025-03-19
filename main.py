import os
import faiss
import openai
import numpy as np
import pickle  # For saving metadata
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
