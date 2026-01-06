import streamlit as st
import pickle
from utils import predict_sentiment

st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1em;
        padding: 0.6em;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1557a0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        padding: 2em 0 1em 0;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3em;
    }
    .footer-name {
        font-weight: 600;
        color: #1f77b4;
        font-size: 1.1em;
    }
    .footer-org {
        color: #888;
        font-size: 0.95em;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with custom styling
st.markdown('<h1 class="main-title">📝 Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type an opinion and get the predicted sentiment instantly</p>', unsafe_allow_html=True)

@st.cache_resource
def load_artifacts(model_path="sentiment_model.pkl", vec_path="vectorizer.pkl"):
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    return vec, clf

vectorizer, classifier = load_artifacts()

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Textbox for user input
user_input = st.text_area(
    "Your opinion:",
    height=140,
    placeholder="e.g., I love how fast this app is.",
    label_visibility="visible"
)

# Button to analyze
if st.button("🔍 Analyze Sentiment"):
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            label = predict_sentiment(text, vectorizer, classifier)
        st.success(f"✨ Predicted sentiment: **{label}**")

# Developer info with better styling
st.markdown("""
    <div class="footer">
        <div class="footer-name">Developed By: Nischal Bhandari</div>
        <div class="footer-org">School of Engineering (SOE), Pokhara University (PU)</div>
    </div>
""", unsafe_allow_html=True)