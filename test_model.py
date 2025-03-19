import torch
from transformers import BertTokenizer, BertModel
import evaluate  # Correct library for BLEU
import numpy as np

# Load BLEU metric from evaluate
bleu = evaluate.load("bleu")

def compute_bert_score(predictions, references):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    model.eval()
    with torch.no_grad():
        pred_tokens = tokenizer(predictions, padding=True, truncation=True, return_tensors="pt")
        ref_tokens = tokenizer(references, padding=True, truncation=True, return_tensors="pt")
        
        pred_embeddings = model(**pred_tokens).last_hidden_state.mean(dim=1)
        ref_embeddings = model(**ref_tokens).last_hidden_state.mean(dim=1)
        
        similarity = torch.nn.functional.cosine_similarity(pred_embeddings, ref_embeddings)
        return similarity.mean().item()

def compute_bleu_score(predictions, references):
    """
    Compute BLEU score using sentence-level input (not tokenized words).
    """
    references = [[ref] for ref in references]  # Wrap each reference in a list
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    return bleu_score

if __name__ == "__main__":
    test_cases = [
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants convert sunlight into energy."),
        ("What are Newton's laws of motion?", "Newton's laws describe the motion of objects: the law of inertia, F=ma, and action-reaction."),
        ("What is the structure of an atom?", "An atom consists of a nucleus containing protons and neutrons, with electrons orbiting around it."),
        ("What is science?", "Science is a way of thinking, observing, and doing things to understand the world we live in and to uncover the secrets of the universe."),
        ("Why is curiosity important in science?", "Curiosity drives scientific discovery by encouraging us to ask questions, explore, and understand the world around us."),
        ("How does science help us understand the world?", "Science allows us to explore and discover how things work, from studying tiny grains of sand to understanding vast galaxies."),
        ("What is the scientific method?", "The scientific method is a step-by-step process involving observation, hypothesis formation, experimentation, and analysis to find answers to questions."),
        ("Give an example of the scientific method in daily life.", "If a pen stops writing, we may hypothesize that the ink has run out, test it by opening the pen, and verify whether the hypothesis is correct."),
        ("Why do we need food to grow?", "Food provides the necessary nutrients and energy for growth and survival."),
        ("How does water change states?", "Water freezes into ice when cooled, turns into steam when heated, and can change states based on temperature."),
        ("Why is Earth unique among planets?", "Earth is the only known planet that supports life due to its environment and water availability."),
        ("How do scientists work?", "Scientists follow the scientific method, conduct experiments, and work together in teams to solve problems and discover new things."),
        ("Why is it important to observe our surroundings?", "Observing our surroundings helps us understand natural phenomena, solve problems, and make scientific discoveries."),
    ]
    
    model_responses = [
    "Photosynthesis is the process in which green plants, algae, and some bacteria use sunlight to convert carbon dioxide and water into glucose and oxygen. This occurs in the chloroplasts of plant cells, where chlorophyll captures light energy.",
    "Newton's laws of motion describe how objects move: the first law is the law of inertia, the second states force equals mass times acceleration (F=ma), and the third states that every action has an equal and opposite reaction.",
    "An atom consists of a central nucleus containing protons and neutrons, with electrons moving around it in orbitals.",
    "Science is a systematic way of exploring and understanding the natural world through observation, experimentation, and reasoning.",
    "Curiosity is important in science because it encourages people to ask questions, seek answers, and explore new ideas, leading to discoveries and advancements.",
    "Science helps us understand the world by explaining natural phenomena, from the smallest particles to vast galaxies, using systematic observation and experimentation.",
    "The scientific method is a structured process that includes observation, forming a hypothesis, conducting experiments, analyzing results, and drawing conclusions.",
    "An example of the scientific method in daily life is checking if a pen has run out of ink by writing with it, then testing a different pen to confirm if the issue is with the ink or the pen itself.",
    "We need food to grow because it provides essential nutrients and energy for the body’s development, repair, and overall functioning.",
    "Water changes states through physical processes: it freezes into ice at low temperatures, melts back to liquid when heated, and evaporates into steam at high temperatures.",
    "Earth is unique among planets because it has liquid water, a stable atmosphere, and conditions that support life.",
    "Scientists follow the scientific method, conduct experiments, and collaborate with others to test hypotheses, analyze data, and discover new knowledge.",
    "Observing our surroundings helps us understand nature, recognize patterns, solve problems, and make scientific discoveries."
    ]
    
    ground_truths = [item[1] for item in test_cases]
    
    bert_score = compute_bert_score(model_responses, ground_truths)
    bleu_score = compute_bleu_score(model_responses, ground_truths)
    
    print(f"BERTScore: {bert_score:.4f}")
    print(f"BLEU Score: {bleu_score:.4f}")
