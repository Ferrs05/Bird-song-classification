import streamlit as st
import numpy as np
import librosa
import joblib
from src.features import extract_mfcc, extract_time_domain, extract_freq_domain

# Load model and scaler
model, scaler = joblib.load('model_rf.pkl')

st.title('Bird Song Classification')

uploaded_file = st.file_uploader('Upload a bird song WAV file', type=['wav'])
if uploaded_file is not None:
    signal, sr = librosa.load(uploaded_file, sr=None)
    # extract features
    mfcc = extract_mfcc(signal, sr)
    td = extract_time_domain(signal)
    fd = extract_freq_domain(signal, sr)
    feat = np.concatenate([mfcc, td, fd]).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = model.predict(feat_scaled)[0]
    st.write(f"Predicted species: **{pred}**")
