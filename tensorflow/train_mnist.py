from calendar import EPOCH
import tensorflow as tf
import matplotlib.pylab as plt

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
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

metrics = ['loss', 'accuracy']
plt.figure()
for i in range(len(metrics)):
 
    metric = metrics[i]
 
    plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
    plt.title(metric)  # グラフのタイトルを表示
    
    plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
    plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
    plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
    plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
    plt.legend()  # ラベルの表示
    
plt.show()  # グラフの表示