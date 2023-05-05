from pyfed.experiment import FL_Experiment
import tensorflow as tf
from sklearn.datasets import fetch_openml
import numpy as np
import copy

def fetch_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X, y = np.array(mnist["data"]), np.array(mnist["target"], dtype='int16')
    return X, y

def distribute_data(X, y, num_clients):
    data_count = len(y) // (num_clients + 1)

    clients_data = []
    clients_target = []
    for i in range(num_clients):
        client_i_X = copy.deepcopy(X[data_count*i:data_count *(i + 1)])
        client_i_y = copy.deepcopy(y[data_count*i:data_count*(i + 1)])

        clients_data.append(client_i_X)
        clients_target.append(client_i_y)

    server_data, server_target = X[data_count *
                                   num_clients:], y[data_count*num_clients:]

    return clients_data, clients_target, server_data, server_target

if __name__ == "__main__":
    lr = 3e-4
    num_clients = 3
    rounds = 2
    epochs = 5
    batch_size = 32
    port = 54321

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

    model.compile(loss=loss,
                optimizer=optimizer(lr),
                metrics=metrics)

    print("\n⏳ Downloading dataset...\n")
    data, target = fetch_mnist()
    print("\n📨 Distributing dataset...\n")
    clients_data, clients_target, server_data, server_target = distribute_data(data, target, num_clients)

    exp = FL_Experiment(num_clients=num_clients,
                        clients_data=clients_data,
                        clients_target=clients_target,
                        server_data=server_data,
                        server_target=server_target)

    exp.run(model=model,
            rounds = rounds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)