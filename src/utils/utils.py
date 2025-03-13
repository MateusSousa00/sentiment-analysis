import re

neutral_phrases = [
    r"\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(year|month|week|day|time|clock|minute|hour)\b",
    r"\b(weather|temperature|forecast|climate|season|sunny|rainy|cloudy|stormy)\b",
    r"\b(who|what|when|where|why|how)\b",
    r"^\d+$",
    r"^\b(i guess|maybe|i think|probably|i feel like|i assume| not sure)\b"

]

def is_textual_input(text):
    """Checks if the input is meaningful text (not just numbers or gibberish)."""
    return bool(re.search(r"[a-zA-Z]", text))

def is_question(text):
    """Detects if the sentence is a question more effectively."""
    return text.strip().endswith("?") or bool(re.match(r"^(who|what|when|where|why|how|do|does|is|are|can|could|should|would|will)\b", text.lower()))

def is_neutral_statement(text):
    """Detects if a phrase is neutral based on common patterns."""
    return any(re.search(pattern, text.lower()) for pattern in neutral_phrases)
