import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
from disp import plot_history, disp_acc

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

    # dropout
    dpt_model = keras.models.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    dpt_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )
    dpt_model_history = dpt_model.fit(
        train_data, train_labels,
        epochs=20, 
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )

    plot_history([
        ('baseline', baseline_history),
        ('dropout', dpt_model_history)
    ])
    metrics = ['loss', 'accuracy']
    disp_acc(result=dpt_model_history, metrics=metrics)


if __name__ == '__main__':
    main()