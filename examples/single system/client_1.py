from pyfed.components import FL_Client
import numpy as np
import tensorflow as tf

epochs = 5
batch_size = 32
lr = 3e-4

loss = "sparse_categorical_crossentropy"
optimizer = tf.optimizers.Adam
metrics = ["accuracy"]

data = np.load("./data_client_1.npy")
target = np.load("./target_client_1.npy")

client1 = FL_Client(name="client_1",
                    data=data,
                    target=target)

client1.train(epochs, batch_size, lr, loss, optimizer, metrics)