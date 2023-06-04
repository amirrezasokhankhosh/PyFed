Introduction
============
PyFed is a compact federated learning framework for TensorFlow models which 
can implement FL algorithms in multiple ways. It is designed to facilitate researchers of distinct 
areas to experiment with FL and utilize it effortlessly.

I recommend that first-time users start by going through the materials below in order to grasp the idea 
behind Federated Learning and how it works in theory.

After such readings, check out the :doc:`Installation` section and start your journey!

What is Federated Learning?
---------------------------
**Deep Learning**

Deep Learning is a method among Machine Learning techniques initially inspired by the human brain. 
A Deep Learning algorithm consists of a Deep Neural Network, a Neural Network with many hidden layers. 
Deep Learning algorithms process elaborate datasets and achieve near-perfect performances in burdensome 
tasks in Computer Vision, Natural Language Processing, and Signal Processing.

**Traditional Training of DNNs**

The traditional approach to solving machine learning problems is centralized, 
meaning that a team of data scientists gather all the data they need, sometimes from different 
organizations, preprocess the aggregated dataset, and train their predefined model on their dataset. 
This approach raises serious security issues and vulnerability for the organizations holding valuable 
information. To illustrate, a team of data scientists might hope to use their knowledge to detect credit 
card fraud across many banks. Hence, each bank must provide the team with its customers' transaction 
information. Needless to say, this brings about security problems, and banks might not trust the team 
with their confidential information. So, what can be done about it?

**Federated Learning**

Imagine if the data scientists could accomplish this task without any direct access to any dataset. Is it possible to 
train a DNN without combining every transaction from every bank and creating an aggregated dataset?

This is the primary motive behind Federated Learning. Usually this can be done in the following steps:

.. note::
    The following FL algorithm is called FedAvg. There are many other algorithms for FL; however, PyFed is now implemented using FedAvg only.

1. Data scientists define a DNN and send it to each organization in their experiment.

2. Each organization now trains the received model with their local data and sends back the trained weights.

3. Then, Data scientists, without knowing what the data was or what those numbers mean, average the received weights and reach a new model.

4. This cycle is repeated until the model has reached satisfactory accuracy.

What is PyFed?
--------------
Pyfed is a federated learning framework that trains any TensorFlow model using the FedAvg algorithm. 
Using it is quite simple and requires no additional information. Anyone with a rudimentary grasp of Machine Learning and Federated Learning
can utilize PyFed.

Pyfed implements FL in multiple ways:

1. It can implement FL using distinct processes in one single machine. You can achieve this by defining each client and the server yourself, or allowing FL Experiment to do it.

2. In addition, and this is where it shines, it can implement FL across multiple systems. The only limitation is that the involved systems must connect to the same network connection.

