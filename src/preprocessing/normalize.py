import re
import unicodedata

def normalize_text(text: str) -> str:
    # Unicode Normalization
    text = unicodedata.normalize("NFC", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove unwanted punctuation (optional)
    text = re.sub(r"[\"\'!.,;:?\-()]+", "", text)

    return text

# Example usage
if __name__ == "__main__":
#     sample = "  नमस्ते! आप कैसे हैं?   "
#     print("Original:", sample)
#     print("Normalized:", normalize_text(sample))

    # print("Script is running...")
    sample = "  नमस्ते! आप कैसे हैं?   "
    print("Original:", sample)
    normalized = normalize_text(sample)
    print("Normalized:", normalized)
    print("Script finished.")
