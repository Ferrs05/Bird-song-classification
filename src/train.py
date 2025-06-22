# src/train.py

import os
import pandas as pd
import joblib
from src.features import get_feature_vector
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model

def main():
    # Baca manifest
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df = pd.read_csv(os.path.join(root, "metadata.csv"))

    # Pisah train/test
    df_train = df[df["split"] == "train"]
    df_test  = df[df["split"] == "test"]

    # Load fitur & label
    def load_subset(df_subset):
        X, y = [], []
        for _, row in df_subset.iterrows():
            subdir = "augmented" if row["type"]=="augmented" else "raw"
            path = os.path.join(root, "data", subdir, row["filename"])
            X.append(get_feature_vector(path))
            y.append(row["label"])
        return X, y

    X_train, y_train = load_subset(df_train)
    X_test,  y_test  = load_subset(df_test)

    # Scaling
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train RF baseline
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)
    joblib.dump((rf, scaler), os.path.join(root, "model_rf.pkl"))

    # Evaluate
    evaluate_model(rf, scaler, X_test_s, y_test)

if __name__ == "__main__":
    main()
