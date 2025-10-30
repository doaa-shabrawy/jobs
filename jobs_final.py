import os
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Nadam
import pickle
import re
import logging
from io import BytesIO
import warnings
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Job Title Predictor',
    layout='wide',
    initial_sidebar_state='auto'
)

# ------------------------- Background -------------------------
def set_background_image(image_path: str):
    if not Path(image_path).exists():
        return
    with open(image_path, "rb") as f:
        encoded = f.read()
    b64 = base64.b64encode(encoded).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0);
        z-index: -1;
    }}
    .app-backdrop {{
        background: rgba(255,255,255,0.85);
        padding: 12px;
        border-radius: 8px;
        position: relative;
        z-index: 10;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------------- Cache -------------------------
try:
    cache_resource = st.cache_resource
except Exception:
    cache_resource = st.cache(allow_output_mutation=True)

# ------------------------- GitHub Downloader -------------------------
@cache_resource
def download_file_if_missing(file_name, github_username, repo_name, folder='models'):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    if not os.path.exists(file_path):
        url = f'https://raw.githubusercontent.com/{github_username}/{repo_name}/main/{folder}/{file_name}'
        r = requests.get(url)
        if r.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(r.content)
        else:
            st.error(f"Failed to download {file_name} from GitHub. Status code: {r.status_code}")
            return None
    return file_path

# ------------------------- Load Artifacts -------------------------
@cache_resource
def load_artifacts(model_name):
    github_username = '<your-username>'
    repo_name = 'job'

    # Ensure tokenizer and label_encoder exist
    tokenizer_path = download_file_if_missing('tokenizer.pkl', github_username, repo_name, folder='.')
    label_encoder_path = download_file_if_missing('label_encoder.pkl', github_username, repo_name, folder='.')
    if not tokenizer_path or not label_encoder_path:
        return None, None, None

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Download model if missing
    model_path = download_file_if_missing(model_name, github_username, repo_name, folder='models')
    if not model_path:
        return None, None, None

    model = load_model(model_path, compile=False)
    optimizer = Nadam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return tokenizer, label_encoder, model

# ------------------------- Text Processing -------------------------
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'job description[a-z]*', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_top_k(model, tokenizer, label_encoder, text, max_len=200, k=3):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    top_idx = probs.argsort()[::-1][:k]
    labels = label_encoder.inverse_transform(top_idx)
    scores = probs[top_idx]
    return list(labels), list(scores)

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# ------------------------- Main App -------------------------
def main():
    set_background_image("pexels-ruslan-burlaka-40570-140945.jpg")
    st.markdown('<div class="app-backdrop">', unsafe_allow_html=True)
    st.title('Job Title Prediction from Description')
    st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.header('Options & Info')

    # List all .keras models from the local folder
    local_models = ['RNN.keras', 'LSTM.keras', 'GRU.keras', 'Embed_Bi_LSTM.keras']
    selected_model = st.sidebar.selectbox('Choose model', local_models)

    with st.spinner(f'Loading model: {selected_model}'):
        tokenizer, label_encoder, model = load_artifacts(selected_model)
        if model is None:
            return

    st.sidebar.write(f'Model: {selected_model}')
    st.sidebar.write(f'Classes: {len(label_encoder.classes_)}')
    top_k = st.sidebar.slider('Show top K predictions', 1, 10, 3)

    example = st.sidebar.selectbox('Pick a sample description', [
        '',
        'We are looking for a software engineer with experience in Python, TensorFlow, and REST APIs.',
        'Seeking a marketing manager to lead product campaigns and manage social media.',
        'Experienced data scientist required: SQL, Python, machine learning models, production deployment.',
        'Administrative assistant needed for scheduling, record keeping, and customer support.'
    ])
    description = st.text_area('Paste your job description here:', value=example, height=220)

    if st.button('Predict Job Title'):
        if not description or len(description.strip()) < 10:
            st.warning('Please paste a longer job description (at least ~10 characters).')
        else:
            cleaned = clean_text(description)
            labels, scores = predict_top_k(model, tokenizer, label_encoder, cleaned, k=top_k)

            col1, col2 = st.columns(2)
            with col1:
                if labels:
                    st.success(f'Primary prediction: {labels[0]} — {scores[0]*100:.2f}% confidence')
                results_df = pd.DataFrame({
                    'Job Category': labels,
                    'Confidence': [f'{s*100:.2f}%' for s in scores]
                })
                st.table(results_df)

            with col2:
                if labels and scores:
                    fig = px.pie(values=[s*100 for s in scores], names=labels, title='Prediction Confidence Distribution', hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader('Key Terms Visualization')
            wc_fig = create_wordcloud(cleaned)
            st.pyplot(wc_fig)

if __name__ == '__main__':
    main()
