import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(10000, input_shape=(1000,)))
  model.add(tf.keras.layers.Dense(10000))
  model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', optimizer='sgd')
print(model.summary())

dataset = tf.data.Dataset.from_tensors(([1.] * 1000, [1.])).repeat(1000).batch(1000)
import time

start = time.time()
model.fit(dataset, epochs=10)
end = time.time()
print(end - start)
