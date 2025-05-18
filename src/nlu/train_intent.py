import json
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from src.preprocessing.normalize import normalize_text

class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_len=32):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len

        self.texts = [normalize_text(item["text"]) for item in data]
        self.labels = label_encoder.transform([item["label"] for item in data])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding["input_ids"].squeeze(),
            'attention_mask': encoding["attention_mask"].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

def load_data(path="data/intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    model_name = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = load_data()
    random.shuffle(data)

    label_encoder = LabelEncoder()
    label_encoder.fit([item["label"] for item in data])

    # Save label encoder mapping
    label_map_path = "src/model/intent_classifier/label_map.json"
    model_output_path = "src/model/intent_classifier/"

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))), f, ensure_ascii=False)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = IntentDataset(train_data, tokenizer, label_encoder)
    val_dataset = IntentDataset(val_data, tokenizer, label_encoder)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    training_args = TrainingArguments(
        output_dir=model_output_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        logging_dir="logs",
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model(model_output_path)

if __name__ == "__main__":
    main()
