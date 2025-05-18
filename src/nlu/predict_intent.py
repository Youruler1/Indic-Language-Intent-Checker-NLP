import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing.normalize import normalize_text
import numpy as np

def load_label_map(path="src/model/intent_classifier/label_map.json"):
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}

def predict_intent(text: str):
    model_dir = "src/model/intent_classifier"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    label_map = load_label_map()

    norm_text = normalize_text(text)
    inputs = tokenizer(norm_text, return_tensors="pt", truncation=True, padding=True, max_length=32)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return label_map[pred], float(torch.max(probs).item())

# Example usage
if __name__ == "__main__":
    while True:
        text = input("You: ")
        intent, confidence = predict_intent(text)
        print(f"Predicted intent: {intent} ({confidence:.2f})\n")
