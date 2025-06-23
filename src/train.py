import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.features import get_feature_vector
from src.evaluate import evaluate_model

def main():
    # Tentukan direktori root proyek
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_file = os.path.join(root, 'bird_songs_metadata.csv')

    # Baca metadata
    df = pd.read_csv(csv_file)
    print('Metadata columns:', df.columns.tolist())

    # Tentukan kolom filename dan label
    filename_col = 'filename' if 'filename' in df.columns else df.columns[0]
    label_col    = 'label'    if 'label'    in df.columns else df.columns[1]

    # Buat split jika belum ada kolom split
    if 'split' not in df.columns:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=0.2,
            stratify=df[label_col],
            random_state=42
        )
        df.loc[train_idx, 'split'] = 'train'
        df.loc[test_idx,  'split'] = 'test'

    # Pisah dataframe
    df_train = df[df['split'] == 'train']
    df_test  = df[df['split'] == 'test']

    # Fungsi load data
    def load_data(sub_df):
        X, y = [], []
        for _, row in sub_df.iterrows():
            fname = row[filename_col]
            label = row[label_col]
            folder = 'augmented' if row.get('type')=='augmented' else 'raw'
            path = os.path.join(root, 'data', folder, fname)
            if not os.path.exists(path):
                print(f"File not found, skipped: {path}")
                continue
            feat = get_feature_vector(path)
            X.append(feat)
            y.append(label)
        return np.array(X), np.array(y)

    # Load fitur train dan test
    X_train, y_train = load_data(df_train)
    X_test,  y_test  = load_data(df_test)

    if len(X_train) == 0:
        raise RuntimeError('Tidak ada data training.')

    # Scaling
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test) if len(X_test) else np.empty((0, X_train_s.shape[1]))

    # Latih Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)

    # Simpan model dan scaler
    joblib.dump((rf, scaler), os.path.join(root, 'model_rf.pkl'))
    print('Model tersimpan di model_rf.pkl')

    # Evaluasi
    if len(X_test):
        evaluate_model(rf, scaler, X_test_s, y_test)
    else:
        print('Tidak ada data test untuk evaluasi')

if __name__ == '__main__':
    main()