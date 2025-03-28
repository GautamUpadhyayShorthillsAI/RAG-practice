import os
import time
import logging
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Setup logging
log_file = "app.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)

class FAISSRAG:
    def __init__(self, index_path="faiss_index.faiss", metadata_path="metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            logger.error("Missing Gemini API key.")
            raise ValueError("Missing Gemini API key. Set it as an environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = GenerativeModel("gemini-1.5-flash")
        self.index, self.metadata = self.load_faiss_index()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a 768-dimension embedding using Gemini AI"""
        logger.info("Generating embedding using Gemini AI.")
        
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="semantic_similarity"
        )
        
        embedding = np.array(response["embedding"], dtype=np.float32)  # Ensure float32 for FAISS
        logger.info("Generated embedding of shape: %s", embedding.shape)
        
        return embedding

    def initialize_faiss_index(self, dimension: int = 768):
        """Initialize a FAISS index with 768-dimension vectors"""
        logger.info("Initializing FAISS index with dimension %d", dimension)
        return faiss.IndexFlatL2(dimension)

    def save_faiss_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("FAISS index and metadata saved successfully.")

    def load_faiss_index(self):
        """Load FAISS index and metadata if available"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info("FAISS index loaded successfully.")
            return index, metadata
        logger.warning("FAISS index not found, creating a new one.")
        return self.initialize_faiss_index(768), []

    def build_index(self, data_folder="data"):
        """Build FAISS index from chunked text files"""
        logger.info("Building FAISS index from chunked text files in %s", data_folder)
        chunks, filenames = self.read_chunks_from_folder(data_folder)

        if not chunks:
            logger.error("No data found to build the FAISS index.")
            return

        self.index = self.initialize_faiss_index(768)  # Ensure correct dimension
        self.metadata = []

        embeddings = []
        for chunk, filename in zip(chunks, filenames):
            embedding = self.generate_embedding(chunk)
            embeddings.append(embedding)
            self.metadata.append({"filename": filename, "text": chunk})  # Store text + filename
            time.sleep(0.25)  # Prevent exceeding API rate limit

        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)

        self.save_faiss_index()
        logger.info("FAISS index built and saved successfully with %d embeddings.", len(chunks))

    def read_chunks_from_folder(self, folder_path: str):
        """Read text chunks from files in a folder"""
        logger.info("Reading chunked text files from %s", folder_path)
        chunks, filenames = [], []
        
        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk = f.read().strip()
                    if chunk:
                        chunks.append(chunk)
                        filenames.append(file)  # Store file name as metadata
        logger.info("Loaded %d chunks from %s", len(chunks), folder_path)
        return chunks, filenames

    def search_faiss(self, query_embedding: np.ndarray, k: int = 5):
        """Search FAISS index for top-k similar embeddings"""
        logger.info("Searching FAISS index with k=%d", k)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return indices[0]

    def query_llm(self, question: str, context: str) -> str:
        prompt = (
            f"Answer the following question strictly based on the provided context. "
            f"If the answer is not in the context, reply with 'I don't know'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        response = self.model.generate_content(prompt)
        return response.text if response else "I don't know."


if __name__ == "__main__":
    logger.info("Starting FAISS index building process...")
    rag = FAISSRAG()
    rag.build_index("data")  # Build FAISS index
    logger.info("FAISS index building completed.")
 