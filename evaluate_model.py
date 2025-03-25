# import os
# import pandas as pd
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
# from datasets import Dataset

# # Load environment variables
# load_dotenv()

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Load CSV file containing questions and expected answers
# df = pd.read_csv("test_cases.csv")

# # Ensure required columns exist
# if "questions" not in df.columns or "expected_answer" not in df.columns:
#     raise ValueError("CSV file must contain 'questions' and 'expected_answer' columns.")

# # Load FAISS index
# faiss_index_path = "faiss_index.faiss"
# index = faiss.read_index(faiss_index_path)

# # Load embedding model (must match the one used for indexing)
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Function to retrieve top-k relevant chunks from FAISS index
# def retrieve_context(question, top_k=3):
#     question_embedding = embedding_model.encode(question).astype(np.float32)
#     _, indices = index.search(np.array([question_embedding]), k=top_k)  # Retrieve top-k chunks
#     retrieved_texts = [df.iloc[idx]["expected_answer"] for idx in indices[0] if idx < len(df)]
#     return " ".join(retrieved_texts) if retrieved_texts else "No relevant context found."

# # Function to query the RAG chatbot
# def get_rag_response(question, context):
#     response = client.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer."},
#             {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
#         ]
#     )
#     return response.choices[0].message.content

# # Retrieve context from FAISS and generate RAG responses
# df["retrieved_context"] = df["questions"].apply(retrieve_context)
# df["generated_answer"] = df.apply(lambda row: get_rag_response(row["questions"], row["retrieved_context"]), axis=1)

# # Convert DataFrame to Hugging Face Dataset format required by Ragas
# dataset = Dataset.from_pandas(df)

# # Evaluate with Ragas
# result = evaluate(
#     dataset=dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
# )

# # Print evaluation results
# print("Ragas Evaluation Results:")
# print(result)

# # Save the evaluation results
# df.to_csv("evaluated_results.csv", index=False)
# print("Evaluation results saved to evaluated_results.csv")


import pandas as pd
import torch
from bert_score import score
from main import FAISSRAG  # Assuming your class is in FAISSRAG.py

# Load CSV file
csv_path = "merged_file.csv"  # Update with your actual file path
df = pd.read_csv(csv_path)

# Initialize FAISS-based RAG system
rag = FAISSRAG()

# Store results
generated_answers = []
expected_answers = df["expected_answer"].tolist()

# Generate answers using RAG
for query in df["question"]:
    query_embedding = rag.generate_embedding(query)
    retrieved_indices = rag.search_faiss(query_embedding, k=5)
    retrieved_context = [rag.metadata[idx] for idx in retrieved_indices if idx < len(rag.metadata)]
    
    generated_answer = rag.query_llm(query, retrieved_context)
    generated_answers.append(generated_answer)

# Compute BERTScore
P, R, F1 = score(generated_answers, expected_answers, lang="en", device="cuda" if torch.cuda.is_available() else "cpu")

# Print Average Scores
print(f"Average BERTScore Precision: {P.mean().item():.4f}")
print(f"Average BERTScore Recall: {R.mean().item():.4f}")
print(f"Average BERTScore F1-Score: {F1.mean().item():.4f}")

# Store results in CSV
df["generated_answer"] = generated_answers
df["BERTScore_F1"] = F1.tolist()
df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to evaluation_results.csv")
