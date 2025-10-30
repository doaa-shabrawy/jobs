#updated
import base64
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


logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Job Title Predictor',
    layout='wide',
    initial_sidebar_state='auto'
)


def set_background_image(image_path: str):
    if not Path(image_path).exists():
        st.warning(f"Background image not found: {image_path}")
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

try:
    cache_resource = st.cache_resource
except Exception:
    cache_resource = st.cache(allow_output_mutation=True)

@cache_resource
def load_artifacts(model_name='LSTM.keras'):
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    model_path = os.path.join('models', model_name)
    model = load_model(model_path, compile=False)
    optimizer = Nadam(learning_rate=1e-3)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return tokenizer, label_encoder, model

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

def main():
    set_background_image("pexels-ruslan-burlaka-40570-140945.jpg")

    st.markdown('<div class="app-backdrop">', unsafe_allow_html=True)
    st.title('Job Title Prediction from Description')
    st.write('Paste a job description and the model will predict the most likely job title / category.')
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar: model selector
    st.sidebar.header('Options & Info')
    model_choices = sorted([f for f in os.listdir('models') if f.endswith('.keras')])
    selected_model = st.sidebar.selectbox('Choose model', model_choices)

    with st.spinner(f'Loading model: {selected_model}'):
        try:
            tokenizer, label_encoder, model = load_artifacts(selected_model)
        except Exception:
            st.error(f"Failed loading model: {selected_model}")
            return

    st.sidebar.write(f'Model: {selected_model}')
    st.sidebar.write(f'Classes: {len(label_encoder.classes_)}')
    top_k = st.sidebar.slider('Show top K predictions', min_value=1, max_value=10, value=3)

    st.sidebar.markdown('---')
    st.sidebar.markdown('Examples:')
    example = st.sidebar.selectbox('Pick a sample description', [
        '',
        'We are looking for a software engineer with experience in Python, TensorFlow, and REST APIs.',
        'Seeking a marketing manager to lead product campaigns and manage social media.',
        'Experienced data scientist required: SQL, Python, machine learning models, production deployment.',
        'Administrative assistant needed for scheduling, record keeping, and customer support.'
    ])

    st.markdown('<div class="app-backdrop" style="margin-top:15px;">', unsafe_allow_html=True)
    description = st.text_area('Paste your job description here:', value=example, height=220)

    col_left, col_right = st.columns([3,1])
    with col_right:
        if st.button('Clear'):
            description = ''

    if st.button('Predict Job Title'):
        if not description or len(description.strip()) < 10:
            st.warning('Please paste a longer job description (at least ~10 characters).')
        else:
            cleaned = clean_text(description)
            labels, scores = predict_top_k(model, tokenizer, label_encoder, cleaned, max_len=200, k=top_k)

            res_col_left, res_col_right = st.columns(2)
            with res_col_left:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                if labels:
                    st.success(f'Primary prediction: {labels[0]} — {scores[0]*100:.2f}% confidence')
                else:
                    st.error("No prediction returned.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader('Top predictions')
                results = pd.DataFrame({
                    'Job Category': labels,
                    'Confidence': [f'{s*100:.2f}%' for s in scores]
                })
                st.table(results)

            with res_col_right:
                st.subheader('Confidence Distribution')
                if labels and scores:
                    fig = px.pie(
                        values=[s * 100 for s in scores],
                        names=labels,
                        title='Prediction Confidence Distribution',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader('Key Terms Visualization')
            wc_fig = create_wordcloud(cleaned)
            if wc_fig:
                st.pyplot(wc_fig)

            with st.expander("View Processed Text"):
                st.subheader('Description (cleaned)')
                st.write(cleaned)
                st.subheader('Original description')
                st.write(description)

            results_df = pd.DataFrame({
                'label': labels,
                'confidence': [float(s) for s in scores],
                'confidence_pct': [f'{s*100:.2f}%' for s in scores]
            })
            csv_buf = BytesIO()
            results_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)
            st.download_button(
                label='Download Results (CSV)',
                data=csv_buf,
                file_name='job_predictions.csv',
                mime='text/csv'
            )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown(
        '<div class="small">Note: Predictions are based on a trained model and may not be perfect. '
        'Consider this as an assistive suggestion.</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()


