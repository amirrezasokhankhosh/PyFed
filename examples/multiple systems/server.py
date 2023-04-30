from pyfed.components import FL_Server
import numpy as np
import tensorflow as tf


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer((784,)))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(2000, activation='relu'))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


loss = "sparse_categorical_crossentropy"
optimizer = tf.optimizers.Adam
metrics = ["accuracy"]

lr = 3e-4
num_clients = 3
rounds = 2

port = 54321

model.compile(loss=loss,
            optimizer=optimizer(lr),
            metrics=metrics)

data = np.load("./data_server.npy")
target = np.load("./target_server.npy")



server = FL_Server(curr_model=model,
                   num_clients=num_clients,
                   rounds=rounds,
                   port=port,
                   multi_system=True)

server.train()
server.test(data, target, loss, optimizer, lr, metrics)