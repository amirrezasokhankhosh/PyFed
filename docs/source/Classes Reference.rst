Classes Reference
=================

**FL_Client**

FL_Client is a class responsible for representing clients in a federated learning network. An object of 
this class trains any model it receives on its local dataset. It is also able to store the training history 
of each round and demonstrate it using Tensorboard. By default, This class is set to run FL locally, however, 
it can be configured to run FL on multiple machines. To do so, alter server_ip and server_port according to your server.

.. py:class:: FL_Client
    Represents clients in a federated learning network.

    .. py:method:: __init__(name, data, target, server_ip=LOCAL_IP, server_port=PORT)
        Creates a FL_Client object

        :param name: name of the client.
        :type name: str
        :param data: training data for any Tensorflow model
        :param target: training target for any Tensorflow model
        :param server_ip: Optional server IP; Used for running on multiple machines
        :type server_ip: str
        :param server_port: Optional server port; Used for running on multiple machines
        :type server_port: str
        :return: FL_Client object.
        :rtype: FL_Client


    .. py:method:: train(epochs, batch_size, lr, loss, optimizer, metrics)
        Receives the global model; trains it locally; and sends the trained model back to the server.

        :param epochs: number of epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: value of learning rate
        :type lr: float
        :param loss: the loss function used in training
        :param optimizer: the optimizer function used in training
        :param metrics: the metrics given to the fit() function of Tensorflow models
        :type metrics: list[str]
        :rtype: None

-------------

**FL_Server**

In PyFed, FL_Server class is used to represent the server in a federated learning network. An object of 
this class takes the global models as an input, broadcast it among the clients, receives the trained models, 
and continue this loop for the number of rounds specified for it. After training the model, it uses Tensorboard to 
demonstrate the training results of each client per round and can test the model on any given test dataset.

By default, this class runs FL locally and on a single system; however, it can be altered to implement FL 
on multiple machines. For further information, check out :doc:`Tutorials` section.

.. py:class:: FL_Server
    Used to represent the server of a federated learning network

    .. py:method:: __init__(curr_model, num_clients, rounds, port=PORT, multi_system=False)
        Creates a FL_Server object

        :param curr_model: Any Tensorflow model
        :param num_clients: Number of clients participating in the experiment
        :type num_clients: int
        :param rounds: Number of rounds
        :type rounds: int
        :param port: the port on which server runs its socket connection
        :type port: str
        :param multi_system: Whether FL is going to run on multiple systems or note
        :type multi_system: bool
        :return: FL_Server object.
        :rtype: FL_Server
    
    .. py:method:: train()
        In each round, sends the global model to each client, receives the trained models, updates the global model using FedAvg

        :rtype: None

    .. py:method:: test(data, target, loss, metrics)
        Tests the global model on the given dataset

        :param data: test data given to evaluate() function of Tensorflow
        :param target: test target given to evaluate() function of Tensorflow
        :param loss: loss function to evaluate the model
        :param metrics: the metrics given to the evaluate() function of Tensorflow models
        :rtype: None

-----------------

**FL_Experiment**

FL_Experiment can be used to test federated learning for a specific model and dataset as fast as possible. 
This class takes some configurations as its input, runs federated learning with just a few lines of code, 
and reports the results of each client along with the accuracy of the model on the test data. 
This class is for those who simply want to experiment with FL, just as the name suggests.

.. py:class:: FL_Experiment
    Implements FL as fast as possible

    .. py:method:: __init__(num_clients, clients_data, clients_target, server_data, server_target, port=PORT)
        Creates a FL_Experiment object

        :param num_clients: Number of clients participating in the experiment
        :type num_clients: int
        :param clients_data: A list of training data for each client
        :type clients_data: list
        :param clients_target: A list of training target for each client
        :type clients_target: list
        :param server_data: Test data used by the server to test the global model
        :param server_data: Test target used by the server to test the global model
        :param port: the port on which server runs its socket connection
        :type port: str
        :return: FL_Experiment object.
        :rtype: FL_Experiment
    
    .. py:method:: run(model, rounds, epochs, batch_size, lr, optimizer, loss, metrics)
        Trains and tests the global model.

        :param model: Any Tensorflow model
        :param rounds: Number of rounds
        :type rounds: int
        :param epochs: number of epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: value of learning rate
        :type lr: float
        :param loss: the loss function used in training
        :param optimizer: the optimizer function used in training
        :param metrics: the metrics given to the fit() function of Tensorflow models
        :type metrics: list[str]
        :rtype: None
