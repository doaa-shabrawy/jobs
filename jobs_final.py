import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Nadam
import pickle
import base64
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ------------------------- Background Image -------------------------
def set_background_image(image_path):
    if not os.path.exists(image_path):
        st.warning(f"Background image not found: {image_path}")
        return

    with open(image_path, "rb") as f:
        encoded = f.read()
    b64 = base64.b64encode(encoded).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------- Load Artifacts -------------------------
def load_artifacts(selected_model):
    """
    Load tokenizer, label encoder, and Keras .h5 model
    Now models are in repo root, not in a folder
    """
    tokenizer_path = f"{selected_model}_tokenizer.pkl"
    label_encoder_path = f"{selected_model}_label_encoder.pkl"
    model_path = f"{selected_model}.h5"  # directly in root

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load Keras model
    model = load_model(model_path, compile=False)
    optimizer = Nadam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    return tokenizer, label_encoder, model

# ------------------------- Main App -------------------------
def main():
    set_background_image("pexels-ruslan-burlaka-40570-140945.jpg")

    st.title("Job Title Prediction from Description")

    # Sidebar for model selection
    local_models = ["RNN", "LSTM", "GRU", "Embed_Bi_LSTM", "Bidirectional_LSTM"]
    selected_model = st.sidebar.selectbox("Choose model", local_models)

    # Load selected model and artifacts
    with st.spinner(f"Loading model: {selected_model}..."):
        tokenizer, label_encoder, model = load_artifacts(selected_model)
        if model is None:
            st.stop()

    # Input text
    input_text = st.text_area("Enter job description:")

    if st.button("Predict"):
        if not input_text:
            st.warning("Please enter a job description first.")
        else:
            # Tokenize input
            seq = tokenizer.texts_to_sequences([input_text])
            max_len = model.input_shape[1]
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq_padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

            # Predict
            pred = model.predict(seq_padded)
            pred_label = label_encoder.inverse_transform([pred.argmax()])[0]

            st.success(f"Predicted Job Title: **{pred_label}**")

    # Optional: WordCloud example
    if st.checkbox("Show WordCloud Example"):
        wc_fig, ax = plt.subplots()
        wc = WordCloud(width=800, height=400).generate(input_text if input_text else "Job")
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(wc_fig)

if __name__ == "__main__":
    main()
