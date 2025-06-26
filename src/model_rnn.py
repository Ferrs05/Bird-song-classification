import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_model(input_shape, num_classes):
    """
    input_shape: (max_len, n_mfcc)
    num_classes: jumlah label
    """
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
