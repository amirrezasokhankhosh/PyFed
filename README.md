# PyFed

PyFed is an open-source framework for federated learning algorithms. Federated Learning is a subfield of machine learning which trains a global model using one server and multiple clients which contain their separate datasets. 
This approach helps clients with the problems of sharing their local data with a server and the risk of data leakage. PyFed is a straightforward and brief package that allows scientists to try Federated Learning for any model using any dataset. Furthermore, PyFed uses Tensorboard to demonstrate the history of training of each client per round.

PyFed implements FL using sockets, processes, and threads. Simply put, each client will run its particular process and tries to establish a socket connection with the server, which also has its specific process. 
Once initiated, each connection will be handled by one thread of the server's process. Each thread will communicate with its respective client to receive the trained weights per round. 
Once they receive the result of one round, threads will return the weights to the server's process, which will arrive at a new model using the mentioned weights. The server will send the new model to the clients using newly initiated threads.
 
PyFed is mainly based on two classes:
 
-> FL_Server: which represents the server to which clients communicate in a federated learning problem. The __train()__ function of this class handles socket connections and the FL policy.
-> FL_Client: which represents each client in a federated learning network. An object of this class handles training procedure any global model on any local data.

Currently, PyFed is limited to FedAvg as its only federated learning policy; however, we will introduce a broader range of configurations for FL experiments in the coming versions.

# Features
PyFed contains two critical classes: FL_Server and FL_Client, which are responsible for server and client actions in a federated learning problem, respectively. </br>
* __FL_Server.train()__ establishes a socket connections with clients and handles weight averaging. In addition, at the end of all rounds a tensorboard session will be started to reveal the efficancy of each client.
* __FL_Server.test()__ will test the final model on the given test data.
* __FL_Client.train()__ will initiate a training session for the client who runs the command. Each client will train the received model on its local dataset.

# Usage
Utilizing PyFed is effortless and time efficient. Following is an example of using this package for the mnist dataset.
# data.py
This is for distributing data among clients and a server.

    import numpy as np
    from sklearn.datasets import fetch_openml

    num_clients = 3
    mnist = fetch_openml("mnist_784", version=1)
    X, y = np.array(mnist["data"]), np.array(mnist["target"], dtype='int16')
    data_count = len(y) // (num_clients + 1)

    for i in range(num_clients):
        client_i_X, client_i_y = X[data_count*i:data_count*(i + 1)], y[data_count*i:data_count*(i + 1)]
        np.save(f"./data_client_{i+1}.npy", client_i_X)
        np.save(f"./target_client_{i+1}.npy", client_i_y)

    server_i_X, server_i_y = X[data_count*num_clients:], y[data_count*num_clients:]
    np.save(f"./data_server.npy", server_i_X)
    np.save(f"./target_server.npy", server_i_y)


## server.py
    from pyfl import FL_Server
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


    model.compile(loss=loss,
                optimizer=optimizer(lr),
                metrics=metrics)

    data = np.load("./data_server.npy")
    target = np.load("./target_server.npy")


    server = FL_Server(model, num_clients, rounds)
    server.train()
    server.test(data, target, loss, optimizer, lr, metrics)

## client_1.py
    from pyfl import FL_Client
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

    client1 = FL_Client("client_1", data, target)

    client1.train(epochs, batch_size, lr, loss, optimizer, metrics)
## client_2.py
    from pyfl import FL_Client
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
## client_3.py
    from pyfl import FL_Client
    import numpy as np
    import tensorflow as tf

    epochs = 5
    batch_size = 32
    lr = 3e-4

    loss = "sparse_categorical_crossentropy"
    optimizer = tf.optimizers.Adam
    metrics = ["accuracy"]

    data = np.load("./data_client_3.npy")
    target = np.load("./target_client_3.npy")

    client3 = FL_Client("client_3", data, target)

    client3.train(epochs, batch_size, lr, loss, optimizer, metrics)

Now, run each file separatly to get federated learning!


