Introduction
============
PyFed is a compact federated learning framework for TensorFlow models which 
can implement FL algorithms in multiple ways. It is designed to facilitate researches of distinct 
areas to experiment with FL and utilize it effortlessly.

I recommend the first-time users start by going through the materials below in order to grasp 
the idea behind Federated Learning and how it is implemented in theory.

After such readings, check out the :doc:`Installation` section and start your journey!

What is Federated Learning?
---------------------------
**Deep Learning**

Deep Learning is a method among Machine Learning techniques which was initially inspired by 
the human brain. A Deep Learning algorithms consists of a Deep Neural Network, a Neural Network with 
many hidden layers. Deep Learning algorithms process elaborate datasets and achieve high accuracies in 
burdensome tasks in Computer Vision, Natural Language Processing, Signal Processing, and so on.

**Traditional Training of DNNs**

Initially, DNNs were trained after the data scientist gathered all the data they were going to use. They 
would ask organizations holding valuable information for permission to use their databases in order to 
train their elaborate and useful models. Nonetheless, this would allow serious security issues for such organizations. 
To illustrate, a team of data scientists might hope to use their knowledge to detect credit card frauds across 
many banks. To do this, each bank involved in this task needs to provide the team with its customers transactions. 
Needless to say, this brings about security problems and banks might not trust the team with their confidential information. 
So, what can be done about it?

**Federated Learning**

Imagine if the data scientists could accomplish this task without any direct access to any dataset. Is it possible to 
train a DNN without aggregating every transaction from every bank?

This is the idea behind Federated Learning. Usually this can be done in the following steps:

.. note::
    The following FL algorithm is called FedAvg. There are many other algorithms for FL; however, PyFed is now implemented using FedAvg only.

1. Data scientists define a DNN and send it to each organization in their experiment.

2. Each organization now trains the recieved model with their local data and sends back the trained weights.

3. Data scientists, without knowing what the data was or what those numbers mean, averages the recieved weights and reaches a new model.

4. This cycle is repeated until the model has reached the satisfactory accuracy.

What is PyFed?
--------------
Pyfed is a federated learning framework that trains any TensorFlow model using FedAvg algorithm. 
Using it is quite simple and requires no additional information. Anyone with a rudimentary grasp of Machine Learning 
can utilize PyFed.

Pyfed implements FL in multiple ways:

1. It can implement FL using distinct processes in one single machine. You can achieve this by defining each client and the server yourself, or allowing FL Experiment to do it.

2. In addition, and this is where it shines, it can implement FL across multiple systems. The only limitation is that the involved systems must connect to one network connection.

