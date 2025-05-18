import json
from pathlib import Path

def validate_intents(path="data/intents.json"):
    path = Path(path)
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decoding error: {e}")
            return

    if not isinstance(data, list):
        print("❌ Root element must be a list.")
        return

    seen = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"❌ Entry {i} is not a dictionary.")
            continue
        if "text" not in item or "label" not in item:
            print(f"❌ Entry {i} missing 'text' or 'label' key.")
            continue
        if not item["text"].strip() or not item["label"].strip():
            print(f"⚠️ Entry {i} has empty 'text' or 'label'.")
            continue
        key = (item["text"], item["label"])
        if key in seen:
            print(f"⚠️ Duplicate entry at {i}: {item}")
        seen.add(key)

    print(f"✅ Checked {len(data)} entries. All good!")

if __name__ == "__main__":
    validate_intents()
