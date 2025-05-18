from indicnlp.tokenize.indic_tokenize import trivial_tokenize

def tokenize_hi(text: str) -> list:
    return trivial_tokenize(text, lang='hi')

# Example usage
if __name__ == "__main__":
    # text = "नमस्ते आप कैसे हैं"
    # tokens = tokenize_hi(text)
    # print("Tokens:", tokens)

    # print("Script is running...")
    
    text = "नमस्ते आप कैसे हैं"
    print("Original text:", text)
    
    tokens = tokenize_hi(text)
    print("Tokens:", tokens)
    
    print("Number of tokens:", len(tokens))
    
    print("Tokens (one per line):")
    for token in tokens:
        print(f"  - {token}")
    
    print("Script finished.")
