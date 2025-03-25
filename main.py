import os
import logging
import faiss
import openai
import numpy as np
import pickle  # For saving metadata
from dotenv import load_dotenv

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
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("Missing OpenAI API key.")
            raise ValueError("Missing OpenAI API key. Set it as an environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.index, self.metadata = self.load_faiss_index()

    def read_chunks_from_folder(self, folder_path: str):
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

    def generate_embedding(self, text: str) -> np.ndarray:
        logger.info("Generating embedding for a query.")
        response = self.client.embeddings.create(
            model="text-embedding-3-small",  
            input=text
        )
        return np.array(response.data[0].embedding)

    def initialize_faiss_index(self, dimension: int):
        logger.info("Initializing FAISS index with dimension %d", dimension)
        return faiss.IndexFlatL2(dimension)
    
    def save_faiss_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("FAISS index saved successfully.")

    def load_faiss_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
            logger.info("FAISS index loaded successfully.")
            return index, metadata
        logger.warning("FAISS index not found, returning None.")
        return None, None

    def search_faiss(self, query_embedding: np.ndarray, k: int = 5):
        logger.info("Searching FAISS index with k=%d", k)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return indices[0]

    def query_llm(self, query: str, context: list):
    # logger.info("Querying LLM for response.")

    # Convert retrieved context to strings (if dictionaries are present)
        context_str_list = [str(item) if isinstance(item, dict) else item for item in context]
        context_str = "\n\n".join(context_str_list)

        prompt = f"""
        You are a knowledgeable assistant. Given the following context, answer the user's question accurately.

        Context:
        {context_str}

        Question: {query}

        Answer:
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an AI assistant."},
                    {"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    logger.info("Starting FAISS index building process...")
    rag = FAISSRAG()
    rag.build_index("data")
    logger.info("FAISS index building completed.")
