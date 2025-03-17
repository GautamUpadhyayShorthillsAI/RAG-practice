import os
import faiss
import openai
import numpy as np
import pickle
import pytest
import pandas as pd  # ✅ Import pandas to read CSV
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# ✅ Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OpenAI API key. Set it as an environment variable.")

# ✅ Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

# ✅ FAISS index and metadata paths
FAISS_INDEX_PATH = "faiss_index.faiss"
METADATA_PATH = "metadata.pkl"
TEST_CASES_CSV = "test_cases.csv"  # ✅ CSV file path
EMBEDDING_DIM = 1536  # Ensure FAISS index and embeddings match

# ✅ Load test cases from CSV
def load_test_cases(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries

# ✅ Load FAISS index and metadata
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print("🔄 FAISS index loaded successfully!")
        return index, metadata
    raise FileNotFoundError("❌ FAISS index or metadata file not found!")

# ✅ Generate embeddings using OpenAI
def generate_embedding(text: str) -> np.ndarray:
    print(f"🧠 Generating embedding for: {text[:50]}...")
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# ✅ Perform FAISS similarity search
def search_faiss(index, query_embedding, metadata, k=3):
    print("🔍 Performing FAISS search...")

    # Ensure correct shape for FAISS search
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    # Check for dimension mismatch
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"❌ Dimension mismatch: Query embedding has {query_embedding.shape[1]}, but FAISS expects {index.d}")

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k)

    # Retrieve matching text chunks
    retrieved_chunks = [metadata[idx]["text"] for idx in indices[0] if idx < len(metadata)]
    
    return retrieved_chunks

# ✅ Generate final answer using GPT-4
def generate_final_answer(query, retrieved_chunks):
    print("🤖 Querying GPT-4 for final answer...")

    context_str = "\n\n".join(retrieved_chunks)  # Combine retrieved chunks
    prompt = f"""
    You are an AI assistant. Based on the following context, answer the user's question accurately.

    Context:
    {context_str}

    Question: {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# ✅ Load test cases from CSV
test_cases = load_test_cases(TEST_CASES_CSV)

# ✅ Evaluate FAISS retrieval + LLM accuracy
@pytest.mark.parametrize("test_case", test_cases)
def test_faiss_with_llm(test_case):
    index, metadata = load_faiss_index()
    
    query = test_case["question"]
    expected_answer = test_case["expected_answer"]

    # Generate query embedding
    query_embedding = generate_embedding(query)

    # Retrieve top results from FAISS
    retrieved_chunks = search_faiss(index, query_embedding, metadata, k=3)

    # Generate the final answer using LLM
    final_answer = generate_final_answer(query, retrieved_chunks)

    # Compute similarity score
    expected_embedding = generate_embedding(expected_answer)
    final_answer_embedding = generate_embedding(final_answer)
    similarity_score = cosine_similarity(final_answer_embedding.reshape(1, -1), expected_embedding.reshape(1, -1))[0][0]

    # Print results
    print(f"\n🔹 Query: {query}")
    print(f"✅ Expected Answer: {expected_answer}")
    print(f"🤖 Final Answer from GPT-4: {final_answer}")
    print(f"📊 Similarity Score: {similarity_score:.4f}")

    # Ensure similarity is above a threshold
    assert similarity_score > 0.7, f"❌ Low similarity score: {similarity_score:.4f}"

