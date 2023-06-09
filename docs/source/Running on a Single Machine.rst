Running on a Single Machine
===========================
In this method, we create separate files in one system and run each of them simultaneasly to achieve federated learning. In this way, we have more control over each client and its local dataset, as opposed to using FL_Experiment.

**data.py**

This code is for distributing data among clients and a server.

.. code-block::

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

**server.py**

This server first trains the model using 3 clients and then, tests it on its local dataset.

.. code-block::

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
    metrics = ["accuracy"]

    num_clients = 3
    rounds = 2

    data = np.load("./data_server.npy")
    target = np.load("./target_server.npy")


    server = FL_Server(curr_model=model,
                    num_clients=num_clients,
                    rounds=rounds)

    server.train()
    server.test(data, target, loss, metrics)

**client_1.py**

.. code-block::

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

Create client_2.py and client_3.py files just like the file above and change the data and target files to match the correct client.

Now, run the server and the clients' files separately and simultaneously to get federated learning!
