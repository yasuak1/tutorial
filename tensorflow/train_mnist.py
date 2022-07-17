import os, sys
sys.path.append(os.pardir)
from calendar import EPOCH
import tensorflow as tf
import matplotlib.pylab as plt
from disp_acc import disp

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
 
x_train.reshape(60000, 28, 28, 1)
x_test.reshape(10000, 28, 28, 1)

# NORMALIZATION
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
  optimizer='adam', 
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
result = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
metrics = ['loss', 'accuracy']

disp(result, metrics)