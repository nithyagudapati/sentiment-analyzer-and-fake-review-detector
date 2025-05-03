import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and models
with open('sentiment_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

sentiment_model = load_model('sentiment_lstm_best_model.keras')
fake_model = load_model('fake_review_lstm_model.keras')

MAX_LEN = 100  # Must match training

# ⬇ Smaller and Stylish Custom Title
st.markdown("""
    <h3 style='text-align: center; color: #20C997; font-weight: bold;'>
        📝 Amazon Sentiment and Fake Review Analyzer.
    </h3>
""", unsafe_allow_html=True)

# Input box
review = st.text_area("Enter your product review:")

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Sentiment Prediction
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        sentiment_score = sentiment_model.predict(padded)[0][0]
        sentiment = "⭐ Positive Sentiment" if sentiment_score > 0.5 else "😞 Negative Sentiment"

        # Fake Review Detection
        review_length = len(review)
        helpfulness_ratio = 0.0  # default
        score = 5  # assumed
        features = np.array([[review_length, helpfulness_ratio, score]])
        fake_score = fake_model.predict(features)[0][0]
        authenticity = "❌ Fake Review" if fake_score >= 0.5 else "✅ Genuine Review"

        st.subheader("Results:")
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Review Authenticity: {authenticity}")