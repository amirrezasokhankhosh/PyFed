---
layout: default
---

# Third Approach: Implementing FL Using Multiple Systems
PyFed is able to run FL algorithms across distinct systems. To do this, connect all systems to the same wifi network. Then, find the local ip of the computer which you would like to use as the server. And at last, create the following files accordingly to run federated learning across distinct machines.

## server.py
```py
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

# The port which you would like to dedicate to the server.
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
```

## client_1.py
```py
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

# The IP of the computer which runs server.py.
server_ip = "192.168.1.153"
# The port you selected in server.py.
server_port = 54321

client1 = FL_Client(name="client_1",
                    data=data,
                    target=target,
                    server_ip=server_ip,
                    server_port=server_port)

client1.train(epochs, batch_size, lr, loss, optimizer, metrics)
```

__client_2.py__ and __client_3.py__ are just like __client_1.py__ but with different data and on different systems. Run each file at the same time to get federated learning on various systems!

## Files
Exact files of these examples can be found on the [GitHub repository](https://github.com/amirrezasokhankhosh/PyFed) of this package.

[__Back to home page.__](./index.md)