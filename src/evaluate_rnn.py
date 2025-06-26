# src/evaluate_rnn.py
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from src.dl_features import get_mfcc_sequence

def main():
    # muat model + mapping
    model = load_model('model_rnn.h5')
    classes = json.load(open('rnn_classes.json'))

    # load metadata
    import pandas as pd, os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
    df = pd.read_csv(os.path.join(root, 'bird_songs_metadata.csv'))
    df_test = df[df['split']=='test']

    X_test = np.array([
        get_mfcc_sequence(os.path.join(root,'data','raw',f))
        for f in df_test['filename']
    ])
    y_true = [classes.index(l) for l in df_test['label']]

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__=='__main__':
    main()
