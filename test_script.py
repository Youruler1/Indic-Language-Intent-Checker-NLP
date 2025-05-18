# test_script.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

from src.preprocessing.normalize import normalize_text
from src.nlu.predict_intent import predict_intent
from src.dialog_manager.rule_based import get_response
from src.utils.config import MODEL_DIR, LABEL_MAP_PATH, DEVICE

def load_model_and_assets():
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Load label map
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    return model.to(DEVICE), tokenizer, label_map

def main():
    print("🤖: नमस्ते! चैटबॉट में आपका स्वागत है। (exit/quit से बाहर निकलें)\n")

    model, tokenizer, label_map = load_model_and_assets()

    while True:
        user_input = input("👤: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("🤖: अलविदा! आपकी सहायता करके अच्छा लगा।")
            break

        # Step 1: Normalize
        normalized_text = normalize_text(user_input)

        # Step 2: Predict intent
        predicted_label = predict_intent(
            normalized_text, tokenizer, model, label_map, DEVICE
        )

        # Step 3: Get response
        response = get_response(predicted_label)
        print(f"🤖: {response}\n")

if __name__ == "__main__":
    main()
