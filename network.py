import random
import numpy as np
import time 

class Network:

    def __init__(self, sizes):
        """Initializes the network. "sizes" is a list with number of neurons per layer."""
        self.sizes = sizes
        self.num_layers = len(sizes)
        # The whole network has a list of np vectors as biases (vecotr per layer) 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        # The whole network has a list of np matrices as weights (matrix per layer)
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:], sizes[:-1])] 

    def __repr__(self):
        return f"Network({self.sizes})"
    
    def correct_input(self, input):
        '''Cheks if the input vecotr is the correct type and dim.'''
        if len(input) != self.sizes[0]: # Check the length
            return False
        if not isinstance(input, np.ndarray): # Check if it a NP vector (array)
            input = np.array(input)
        return True
    
    def feedforward(self, input, correct=None):
        """Runs the input a through the whole network."""
        if (correct is None) or (correct is False): 
            if not self.correct_input(input):
                return f"Your vector size is {len(input)}, instead of {len(self.sizes[0])}."
        layer_value = input
        for W, b in zip(self.weights, self.biases):
            layer_value = sigma_fun(W.dot(layer_value) + b)
        return layer_value
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None, num=None):
        '''The learning algorythm for the network.'''
        training_data_size = len(training_data)

        if test_data is not None: 
            num = len(test_data)
            accuracy_progression = []

        # Batch the data and run through all the epochs
        for epoch in range(epochs):
            if test_data is not None: 
                start_time = time.time()
            random.shuffle(training_data)
            for i in range(training_data_size // mini_batch_size):
                batch = training_data[i*mini_batch_size : (i+1)*mini_batch_size]
                self.update_mini_batch(batch, learning_rate)
            if test_data is not None: 
                accuracy = self.evaluate(test_data, num)
                epoch_time = time.time() - start_time
                print(f'Epoch {epoch + 1}/{epochs} complete, accuracy is {accuracy}')
                accuracy_progression.append((accuracy, epoch, epoch_time))
            else:
                print(f'Epoch {epoch + 1}/{epochs} complete.')

        if test_data is not None: 
            self.accuracy = accuracy_progression

    def update_mini_batch(self, mini_batch, learning_rate):
        '''Applies gradient descent to the given batch.'''
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_W = [np.zeros(W.shape) for W in self.weights]
        for x, y in mini_batch: 
            delta_gradient_b, delta_gradient_W = self.backprop(x, y)

            # Update the weights and biases
            for i, delta_grad in enumerate(delta_gradient_b):
                gradient_b[i] += delta_grad
            for i, delta_grad in enumerate(delta_gradient_W):
                gradient_W[i] += delta_grad

        # Apply the gradient descent to the network
        for i, elt in enumerate(self.biases):
            self.biases[i] = elt - ((learning_rate / len(mini_batch)) * gradient_b[i])
        for i, elt in enumerate(self.weights):
            self.weights[i] = elt - ((learning_rate / len(mini_batch)) * gradient_W[i])

    def backprop(self, x, y):
        '''Calculates the gradient of the cost function with the backpropagation algorythm.'''
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_W = [np.zeros(W.shape) for W in self.weights]

        # Feedforward (different to the method because we need more information)
        activation = x
        activations = [x] # Each layer stores its activations
        pre_sigmoid_act = []
        for W, b in zip(self.weights, self.biases):
            activation = W.dot(activation) + b
            pre_sigmoid_act.append(activation)
            activation = sigma_fun(activation)
            activations.append(activation)

        # Output layer error
        delta = self.cost_fun_derivative(activations[-1], y) * sigma_fun_derivative(pre_sigmoid_act[-1])
        gradient_W[-1] = delta.dot(activations[-2].transpose())
        gradient_b[-1] = delta

        # Backpropagation through the layers 
        for l in range(2, self.num_layers):
            z = pre_sigmoid_act[-l]
            a = sigma_fun_derivative(z)
            W_T = self.weights[-l+1].transpose()
            delta = W_T.dot(delta) * a
            gradient_W[-l] = delta.dot(activations[-l-1].transpose())
            gradient_b[-l] = delta
        
        return gradient_b, gradient_W

    def cost_fun_derivative(self, out_activations, y):
        '''Derivative of the standard cost function for a single training example.'''
        return out_activations - y
    
    def evaluate(self, test_data, num):
        '''Evaluates the performace of the neural network.'''
        random.shuffle(test_data)
        test_batch = test_data[0:num]

        # The 'guess' of the network is the highes activation in the last layer 
        result = [(np.argmax(self.feedforward(x)), y) for x, y in test_batch]

        return list(sum([y[x] == 1 for x, y in result]) / len(test_batch))[0]


# Other functions
def sigma_fun(x): 
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def sigma_fun_derivative(x):
    # Not stable if the derivate is calculated by hand
    return sigma_fun(x) * (1 - sigma_fun(x)) # Works on vectors
    
