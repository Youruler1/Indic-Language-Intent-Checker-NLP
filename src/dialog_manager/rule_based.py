# src/dialog_manager/rule_based.py

def get_response(intent: str) -> str:
    responses = {
        "greet": "नमस्ते! मैं आपकी कैसे मदद कर सकती हूँ?",
        "bye": "अलविदा! फिर मिलेंगे।",
        "ask_weather": "अभी मौसम की जानकारी के लिए कृपया अपना स्थान बताएं।",
        "ask_time": "अभी समय जानने के लिए मैं आपकी मदद कर सकती हूँ।",
        "ask_bot_info": "मैं एक हिंदी में बात करने वाला चैटबॉट हूँ।",
        "ask_capabilities": "मैं मौसम, समय, और सामान्य प्रश्नों के उत्तर दे सकता हूँ।",
        "thank_you": "आपका स्वागत है! मुझे आपकी मदद करके खुशी हुई।",
        "unknown": "माफ़ कीजिए, मैं आपकी बात नहीं समझ सकी। कृपया दोहराएँ।"
    }

    return responses.get(intent, responses["unknown"])
