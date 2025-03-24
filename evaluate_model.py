import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from datasets import Dataset

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load CSV file containing questions and expected answers
df = pd.read_csv("test_cases.csv")

# Ensure required columns exist
if "questions" not in df.columns or "expected_answer" not in df.columns:
    raise ValueError("CSV file must contain 'questions' and 'expected_answer' columns.")

# Load FAISS index
faiss_index_path = "faiss_index.faiss"
index = faiss.read_index(faiss_index_path)

# Load embedding model (must match the one used for indexing)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to retrieve top-k relevant chunks from FAISS index
def retrieve_context(question, top_k=3):
    question_embedding = embedding_model.encode(question).astype(np.float32)
    _, indices = index.search(np.array([question_embedding]), k=top_k)  # Retrieve top-k chunks
    retrieved_texts = [df.iloc[idx]["expected_answer"] for idx in indices[0] if idx < len(df)]
    return " ".join(retrieved_texts) if retrieved_texts else "No relevant context found."

# Function to query the RAG chatbot
def get_rag_response(question, context):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Retrieve context from FAISS and generate RAG responses
df["retrieved_context"] = df["questions"].apply(retrieve_context)
df["generated_answer"] = df.apply(lambda row: get_rag_response(row["questions"], row["retrieved_context"]), axis=1)

# Convert DataFrame to Hugging Face Dataset format required by Ragas
dataset = Dataset.from_pandas(df)

# Evaluate with Ragas
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
)

# Print evaluation results
print("Ragas Evaluation Results:")
print(result)

# Save the evaluation results
df.to_csv("evaluated_results.csv", index=False)
print("Evaluation results saved to evaluated_results.csv")
