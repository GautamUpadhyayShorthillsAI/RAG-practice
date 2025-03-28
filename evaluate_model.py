import os
import time
import pandas as pd
import torch
import random
from main import FAISSRAG  # Import your RAG class
from sklearn.metrics import precision_score, f1_score
from bert_score import score as bert_score
from google.api_core.exceptions import ResourceExhausted

class RAGEvaluator:
    def __init__(self, csv_file, max_retries=5):
        """Initialize evaluator with test cases and RAG pipeline."""
        self.csv_file = csv_file
        self.test_cases = self.load_test_cases()
        self.rag = FAISSRAG()
        self.max_retries = max_retries  # Maximum retry attempts
    
    def load_test_cases(self):
        """Load test cases from a CSV file."""
        return pd.read_csv(self.csv_file)
    
    def generate_answer(self, question):
        """Generate an answer using the LLM based on retrieved context with retry handling."""
        retries = 0
        while retries < self.max_retries:
            try:
                query_embedding = self.rag.generate_embedding(question)
                indices = self.rag.search_faiss(query_embedding, k=5)
                retrieved_texts = [self.rag.metadata[i]['text'] for i in indices if i < len(self.rag.metadata)]
                context = '\n'.join(retrieved_texts)
                return self.rag.query_llm(question, context)
            
            except ResourceExhausted as e:
                retry_delay = 2 ** retries + random.uniform(0, 1)  # Exponential backoff
                print(f"⚠️ Rate limit reached. Retrying in {retry_delay:.2f} seconds...")
                time.sleep(retry_delay)
                retries += 1
        
        print("❌ Failed to generate an answer after multiple retries.")
        return "API limit exceeded. Please try again later."
    
    def evaluate_rag(self):
        """Evaluate RAG pipeline using BERTScore, Precision, and F1-score."""
        questions = self.test_cases['question'].tolist()
        expected_answers = self.test_cases['expected_answer'].tolist()
        generated_answers = []

        for i, question in enumerate(questions):
            print(f"Processing {i+1}/{len(questions)}: {question[:50]}...")  # Log progress
            generated_answer = self.generate_answer(question)
            generated_answers.append(generated_answer)

            if (i + 1) % 10 == 0:  # Save progress every 10 samples
                pd.DataFrame({
                    "question": questions[:i+1],
                    "generated_answer": generated_answers,
                    "ground_truth": expected_answers[:i+1]
                }).to_csv("partial_evaluation_results.csv", index=False)
                print("✅ Partial results saved.")

        # Compute BERTScore
        P, R, F1 = bert_score(generated_answers, expected_answers, lang="en", model_type="bert-base-uncased")

        # Convert tensor values to lists
        precision_scores = P.tolist()
        f1_scores = F1.tolist()

        # Prepare evaluation dataset
        eval_data = pd.DataFrame({
            "question": questions,
            "generated_answer": generated_answers,
            "ground_truth": expected_answers,
            "bertscore": R.tolist(),
            "precision": precision_scores,
            "f1_score": f1_scores
        })
        
        # Save evaluation results to CSV
        eval_data.to_csv("rag_evaluation_results.csv", index=False)
        print("🎯 Evaluation completed. Results saved to rag_evaluation_results.csv")

        return eval_data

if __name__ == "__main__":
    csv_file = "merged_file.csv"  # Update with your actual file path
    evaluator = RAGEvaluator(csv_file, max_retries=5)  # Set max retries
    evaluator.evaluate_rag()
