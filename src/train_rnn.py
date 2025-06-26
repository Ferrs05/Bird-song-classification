# src/train_rnn.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.dl_features import get_mfcc_sequence
from src.model_rnn import build_rnn_model

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    meta = pd.read_csv(os.path.join(root, 'bird_songs_metadata.csv'))
    # split stratified
    if 'split' not in meta: 
        train_idx, test_idx = train_test_split(
            meta.index, test_size=0.2, stratify=meta['label'], random_state=42)
        meta.loc[train_idx, 'split'] = 'train'
        meta.loc[test_idx,  'split'] = 'test'

    df_train = meta[meta['split']=='train']
    df_test  = meta[meta['split']=='test']

    # map label ke integer
    classes = sorted(df_train['label'].unique())
    label_to_idx = {c:i for i,c in enumerate(classes)}

    # siapkan data
    X_train = np.array([
        get_mfcc_sequence(os.path.join(root,'data','raw',f))
        for f in df_train['filename']
    ])
    y_train = np.array([label_to_idx[l] for l in df_train['label']])

    X_test = np.array([
        get_mfcc_sequence(os.path.join(root,'data','raw',f))
        for f in df_test['filename']
    ])
    y_test = np.array([label_to_idx[l] for l in df_test['label']])

    # build & train
    model = build_rnn_model(input_shape=X_train.shape[1:], num_classes=len(classes))
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('model_rnn.h5', save_best_only=True)
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50, batch_size=32,
        callbacks=callbacks
    )

    # simpan classes mapping
    import json
    with open('rnn_classes.json','w') as f:
        json.dump(classes, f)

if __name__=='__main__':
    main()
