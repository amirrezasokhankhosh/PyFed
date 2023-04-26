from pyfed.pyfed import FL_Client
import numpy as np
import tensorflow as tf

epochs = 5
batch_size = 32
lr = 3e-4

loss = "sparse_categorical_crossentropy"
optimizer = tf.optimizers.Adam
metrics = ["accuracy"]

data = np.load("./data_client_2.npy")
target = np.load("./target_client_2.npy")

client2 = FL_Client("client_2", data, target)

client2.train(epochs, batch_size, lr, loss, optimizer, metrics)