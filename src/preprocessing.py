# src/preprocessing.py

import os
import numpy as np
import librosa
import soundfile as sf

# ======================================================================
# DEFINISI PATH KE PROYEK 
ROOT_DIR = r"C:\Users\muhfe\OneDrive\Documents\BELAJAR\File Project\Project Birdsong"

# Folder data asli (.wav) dan tujuan hasil augmentasi
DATA_FOLDER      = os.path.join(ROOT_DIR, "data", "raw")
AUGMENTED_FOLDER = os.path.join(ROOT_DIR, "data", "augmented")
# ======================================================================

def load_audio(path, sr=22050):
    """
    Load sebuah file audio .wav sebagai waveform numpy array.
    """
    signal, sample_rate = librosa.load(path, sr=sr)
    return signal, sample_rate

def save_audio(signal, sr, dst_path):
    """
    Simpan numpy array audio ke file dst_path.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, signal, sr)

def augment_time_stretch(signal, rate=1.1):
    """
    Percepat/melambatkan tempo audio tanpa mengubah pitch.
    rate > 1.0 = lebih cepat; rate < 1.0 = lebih lambat.
    """
    return librosa.effects.time_stretch(signal, rate)

def augment_pitch_shift(signal, sr, n_steps=2):
    """
    Naik/turunkan pitch audio sejumlah n_steps semitone.
    """
    return librosa.effects.pitch_shift(signal, sr, n_steps)

def augment_noise(signal, noise_factor=0.005):
    """
    Tambahkan noise putih ke audio.
    """
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def create_augmented_dataset():
    """
    Baca semua .wav di DATA_FOLDER, buat augmentasi,
    dan simpan hasilnya (plus original) di AUGMENTED_FOLDER.
    """
    for fname in os.listdir(DATA_FOLDER):
        if not fname.lower().endswith('.wav'):
            continue

        src_path = os.path.join(DATA_FOLDER, fname)
        signal, sr = load_audio(src_path)

        # 1) simpan versi original
        dst_orig = os.path.join(AUGMENTED_FOLDER, f"orig_{fname}")
        save_audio(signal, sr, dst_orig)

        # 2) time stretch
        stretched = augment_time_stretch(signal, rate=0.9)
        dst_stretch = os.path.join(AUGMENTED_FOLDER, f"stretch_{fname}")
        save_audio(stretched, sr, dst_stretch)

        # 3) pitch shift
        pitched = augment_pitch_shift(signal, sr, n_steps=2)
        dst_pitch = os.path.join(AUGMENTED_FOLDER, f"pitch_{fname}")
        save_audio(pitched, sr, dst_pitch)

        # 4) noise injection
        noised = augment_noise(signal, noise_factor=0.005)
        dst_noise = os.path.join(AUGMENTED_FOLDER, f"noise_{fname}")
        save_audio(noised, sr, dst_noise)

if __name__ == "__main__":
    # jalankan augmentasi secara langsung jika dieksekusi sebagai script
    create_augmented_dataset()
