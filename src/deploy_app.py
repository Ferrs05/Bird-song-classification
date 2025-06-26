# src/deploy_app.py
import streamlit as st
import numpy as np
import joblib
import json
import os

import librosa
from src.features      import extract_mfcc, extract_time_domain, extract_freq_domain
from src.dl_features   import get_mfcc_sequence
from tensorflow.keras.models import load_model

# Pilih model
choice = st.sidebar.selectbox("Pilih model", ["Random Forest","RNN"])

if choice == "Random Forest":
    # Load RF model & scaler
    model, scaler = joblib.load(os.path.join("model_rf.pkl"))
    # Load dan ekstrak fitur 1D
    uploaded = st.file_uploader("Upload file WAV", type="wav")
    if uploaded:
        signal, sr = librosa.load(uploaded, sr=None)
        mfcc = extract_mfcc(signal, sr)
        td   = extract_time_domain(signal)
        fd   = extract_freq_domain(signal, sr)
        feat = np.concatenate([mfcc, td, fd]).reshape(1, -1)
        feat_s = scaler.transform(feat)
        pred = model.predict(feat_s)[0]
        st.write(f"Predicted (RF): **{pred}**")

else:
    # Load RNN model & class mapping
    rnn_model    = load_model(os.path.join("model_rnn.h5"))
    classes      = json.load(open("rnn_classes.json"))
    uploaded     = st.file_uploader("Upload file WAV", type="wav")
    if uploaded:
        # Ekstraksi MFCC sequence (padding/truncate ke max_len)
        seq = get_mfcc_sequence(uploaded)        # shape (max_len, n_mfcc)
        seq = seq[np.newaxis, ..., np.newaxis]   # tambah batch & channel
        probs = rnn_model.predict(seq)[0]
        pred_idx = np.argmax(probs)
        pred = classes[pred_idx]
        st.write(f"Predicted (RNN): **{pred}**")
