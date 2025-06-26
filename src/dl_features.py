import numpy as np
import librosa

def get_mfcc_sequence(path, sr=22050, n_mfcc=13, max_len=200):
    """
    Load .wav → hitung MFCC per frame → pad/truncate ke panjang max_len.
    Output shape: (max_len, n_mfcc)
    """
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # shape: (t, n_mfcc)
    if mfcc.shape[0] < max_len:
        pad = np.zeros((max_len - mfcc.shape[0], n_mfcc))
        mfcc = np.vstack([mfcc, pad])
    else:
        mfcc = mfcc[:max_len]
    return mfcc
