import json
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing.normalize import normalize_text
from sklearn.model_selection import train_test_split
import numpy as np

def load_label_encoder(path="src/model/intent_classifier/label_map.json"):
    with open(path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(list(label_map.keys()))
    return encoder, {v: k for k, v in label_map.items()}

def load_data(path="data/intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate():
    model_path = "src/model/intent_classifier"
    label_map_path = f"{model_path}/label_map.json"

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

    label_encoder, id2label = load_label_encoder(label_map_path)

    data = load_data()
    _, test_data = train_test_split(data, test_size=0.2, random_state=42)

    true_labels = []
    pred_labels = []

    for item in test_data:
        text = normalize_text(item["text"])
        true_label = item["label"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

        pred_label = id2label[pred_id]

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    print("Classification Report:\n")
    print(classification_report(true_labels, pred_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

if __name__ == "__main__":
    evaluate()
