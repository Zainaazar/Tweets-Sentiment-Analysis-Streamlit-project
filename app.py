import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

import altair as alt

# st.set_option('deprecation.showfileUploaderEncoding', False)

# ==============================
# 🎯 Load the trained model and tokenizer
# ==============================
model = tf.keras.models.load_model("hybrid_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define label map (same as training)
label_map = {
    0: 'Negative 😞',
    1: 'Positive 😊',
    2: 'Neutral 😐',
    3: 'Irrelevant 🤔'
}

# Load max_len from your training phase

MAX_LEN = 130 

# ==============================
# 🧠 Prediction function
# ==============================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)
    label_index = np.argmax(pred, axis=1)[0]
    sentiment = label_map[label_index]
    return sentiment

# ==============================
# 🎨 Streamlit App Layout
# ==============================
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #4B0082;'>💬 Twitter Sentiment Analysis App</h1>
    <p style='text-align: center;'>Enter a tweet below to analyze its sentiment using a hybrid LSTM-GRU deep learning model.</p>
    """,
    unsafe_allow_html=True
)

# Text input from user
user_input = st.text_area("✍️ Enter a tweet:", placeholder="Type a tweet here...")

# Prediction button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                result = predict_sentiment(user_input)
                st.success(f"**Predicted Sentiment:** {result}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text before analyzing.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by Zaina — Powered by TensorFlow & Streamlit 🚀</p>",
    unsafe_allow_html=True
)
