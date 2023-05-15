# Introduction
PyFed is an open-source framework for federated learning. PyFed is fairly straightforward and brief in comparison to other federated learning frameworks. Furthermore, it allows running federated learning algorithms with any Tensorflow dataset on any preprocessed dataset. PyFed introduces several methods of federated learning implementation such as running multiple processes on a single machine and training on various systems. In addition, PyFed employs Tensorboard to demonstrate the history of training of each client and assess loss and accuracy of each client per round.

PyFed implements FL using sockets, processes, and threads. Simply put, each client will run its particular process and tries to establish a socket connection with the server, which also has its specific process. Once initiated, each connection will be handled by one thread of the server's process. Each thread will communicate with its respective client to receive the trained weights per round. Once they receive the result of one round, threads will return the weights to the server's process, which will arrive at a new model using the mentioned weights. The server will send the new model to the clients using newly initiated threads.

PyFed is mainly based on two classes:

- FL_Server: which represents the server to which clients communicate in a federated learning problem. The train() function of this class handles socket connections and the FL policy.
- FL_Client: which represents each client in a federated learning network. An object of this class handles training procedure of any global model on any local data.

PyFed can run federated learning in 3 ways, the tutorial of each is attached to the following links:

1. [Running FL only on one system and using separate processes (Using FL_Experiment)](./fl_exp.md).
2. [Running FL only on one system and using separate processes (Using FL_Server and FL_Client)](./multi_proc.md).
2. [Running FL on multiple systems](./multi_sys.md).


Currently, PyFed is limited to FedAvg as its only federated learning policy; however, a broader range of configurations for FL experiments will be introduced in the future versions.

# Features
PyFed contains two critical classes: FL_Server and FL_Client, which are responsible for server and client actions in a federated learning problem, respectively. </br>
* __FL_Server.train()__ establishes a socket connections with clients and handles weight averaging. In addition, at the end of all rounds a tensorboard session will be started to reveal the efficancy of each client.
* __FL_Server.test()__ will test the final model on the given test data.
* __FL_Client.train()__ will initiate a training session for the client who runs the command. Each client will train the received model on its local dataset.

# Installation
## MacOS
```
pip install pyfed-macos==0.0.28
```
## Windows, Linux
```
pip install pyfed==0.0.28
```