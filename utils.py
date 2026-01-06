import re

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Lowercase
    text = text.lower()
    # Keep alphanumerics and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text: str, vectorizer, classifier) -> str:
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    return classifier.predict(vec)[0]
