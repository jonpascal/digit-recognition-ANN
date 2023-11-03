import pickle
import random
import numpy as np

from network import Network
import data_loader

training_data = data_loader.read_idx_ubyte("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
testing_data = data_loader.read_idx_ubyte("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

all_data = {}

# ANN with 3 layers and variable sizes
data = {}

for sizes in ([784, 32, 10], [784, 64, 10], [784, 128, 10]): 
    random.seed(12345678)
    np.random.seed(12345678)
    network = Network(sizes)
    network.SGD(training_data, 30, 25, 0.08, testing_data, 100)
    data[f'W_{sizes[0]}.{sizes[1]}.{sizes[2]}_30_25_0.08_nl'] = network.weights
    data[f'b_{sizes[0]}.{sizes[1]}.{sizes[2]}_30_25_0.08_nl'] = network.biases
    data[f'a_{sizes[0]}.{sizes[1]}.{sizes[2]}_30_25_0.08_nl'] = network.accuracy

all_data.update(data)

# ANN with 4 layers and variable sizes
data = {}

for sizes in ([784, 128, 32, 10], [784, 156, 48, 10], [784, 196, 64, 10]): 
    random.seed(12345678)
    np.random.seed(12345678)
    network = Network(sizes)
    network.SGD(training_data, 30, 25, 0.08, testing_data, 100)
    data[f'W_{sizes[0]}.{sizes[1]}.{sizes[2]}.{sizes[3]}_30_25_0.08_nl'] = network.weights
    data[f'b_{sizes[0]}.{sizes[1]}.{sizes[2]}.{sizes[3]}_30_25_0.08_nl'] = network.biases
    data[f'a_{sizes[0]}.{sizes[1]}.{sizes[2]}.{sizes[3]}_30_25_0.08_nl'] = network.accuracy

all_data.update(data)


# ANN with 4 layers and hyper parameters changing
sizes = [784, 96, 48, 10]

# Number of epochs
data = {}

random.seed(12345678)
np.random.seed(12345678)
network = Network(sizes)
network.SGD(training_data, 500, 25, 0.08, training_data, 100)
data[f'W_784.96.48.10_500_25_0.08_tr'] = network.weights
data[f'b_784.96.48.10_500_25_0.08_tr'] = network.biases
data[f'a_784.96.48.10_500_25_0.08_tr'] = network.accuracy

random.seed(12345678)
np.random.seed(12345678)
network = Network(sizes)
network.SGD(training_data, 500, 25, 0.08, testing_data, 100)
data[f'W_784.96.48.10_500_25_0.08_te'] = network.weights
data[f'b_784.96.48.10_500_25_0.08_te'] = network.biases
data[f'a_784.96.48.10_500_25_0.08_te'] = network.accuracy

all_data.update(data)

# Mini-batch size
data = {}

for mini_batch_size in [1, 10, 25, 100]: 
    random.seed(12345678)
    np.random.seed(12345678)
    network = Network(sizes)
    network.SGD(training_data, 30, mini_batch_size, 0.08, testing_data, 100)
    data[f'W_784.96.48.10_30_{mini_batch_size}_0.08_m'] = network.weights
    data[f'b_784.96.48.10_30_{mini_batch_size}_0.08_m'] = network.biases
    data[f'a_784.96.48.10_30_{mini_batch_size}_0.08_m'] = network.accuracy

all_data.update(data)

# Learning rate
data = {}

for learning_rate in [0.01, 0.08, 0.2, 0.8]: 
    random.seed(12345678)
    np.random.seed(12345678)
    network = Network(sizes)
    network.SGD(training_data, 30, 25, learning_rate, training_data, 100)
    data[f'W_784.96.48.10_30_25_{learning_rate}_l'] = network.weights
    data[f'b_784.96.48.10_30_25_{learning_rate}_l'] = network.biases
    data[f'a_784.96.48.10_30_25_{learning_rate}_l'] = network.accuracy

all_data.update(data)


with open("model_parameters.pkl", "wb") as f:
    pickle.dump(all_data, f)

