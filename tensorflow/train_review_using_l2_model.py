import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
from disp import plot_history

def main():
    NUM_WORDS = 10000
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

    def multi_hot_sequences(sequences, dimension):
        results = np.zeros((len(sequences), dimension))
        for i, word_indices in enumerate(sequences):
            results[i, word_indices] = 1.0
        return results
    
    train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

    # base
    baseline_model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    baseline_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
        )

    baseline_model.summary()

    baseline_history = baseline_model.fit(
        train_data, train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

    # l2_model
    l2_model = keras.models.Sequential([
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
        ])
    l2_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )
    l2_model.summary()

    l2_model_history = l2_model.fit(
        train_data, train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

    plot_history([
        ('baseline', baseline_history)
        ('l2', l2_model_history)
    ])

    

if __name__ == '__main__':
    main()