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