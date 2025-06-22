import numpy as np
import librosa


def extract_mfcc(signal, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_time_domain(signal):
    # zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(signal)
    # energy
    energy = np.sum(signal**2) / len(signal)
    return np.array([np.mean(zcr), energy])


def extract_freq_domain(signal, sr):
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    return np.array([np.mean(centroid), np.mean(bandwidth), np.mean(rolloff)])


def get_feature_vector(path):
    signal, sr = librosa.load(path, sr=None)
    mfcc = extract_mfcc(signal, sr)
    td = extract_time_domain(signal)
    fd = extract_freq_domain(signal, sr)
    return np.concatenate([mfcc, td, fd])