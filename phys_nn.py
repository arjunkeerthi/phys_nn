import numpy as np
import copy
from sklearn import datasets
import sklearn.metrics
import time
from progress import ProgressBarThread

class NN:
    # layer_node_count = list of number of nodes in each layers
    # - First element: Number of nodes in input layer, Last: output layer
    # bias = Bias for the corresponding layer counted in layer_node_count
    def __init__(self, layer_node_count=[2,2,2], bias=np.ones(2)):
        self.learning_rate = 0.1

        # define activation function and its derivative
        self.sigmoid = lambda z: 1/(1+np.exp(-z))
        self.sigmoid_derivative = lambda z: self.sigmoid(z)*(1-self.sigmoid(z));

        self.num_layers = len(layer_node_count)

        # instantiate the weights matrices
        np.random.seed(2)
        self.W = []
        for i in range(self.num_layers-1):
            # Add weights between i and i+1 layer
            weights = np.random.rand(layer_node_count[i], layer_node_count[i+1])*2-1
            self.W.append(weights)

        # Should be length of num_layers-1
        self.bias = []
        for i in range(self.num_layers-1):
            biases = np.empty(layer_node_count[i+1])
            biases.fill(bias[i])
            self.bias.append(biases)

        # nodes[i] is a vector of length layer_node_count[i]
        self.nodes = []
        for i in range(self.num_layers):
            layer = np.zeros((layer_node_count[i],1))
            self.nodes.append(layer);

        self.pre_activation_nodes = copy.deepcopy(self.nodes)

    # give the layer value to update
    def feedforward(self, layer):
        # bounds checking
        if layer == 0:
            print("can not feedforward to the input layer")
            return
        if layer >= self.num_layers:
            print("feedforward out of bounds")
            return

        # grab relevant values
        weights = self.W[layer-1]
        prev_node_layer = self.nodes[layer-1]
        layer_bias = self.bias[layer-1]

        # calculate the 'z' from the lecture notes
        activation_input = prev_node_layer@weights+layer_bias
        self.pre_activation_nodes[layer] = activation_input

        # feed forward to the node
        self.nodes[layer] = self.sigmoid(activation_input)

    # Calculate error in output layer
    def error(self, outputs):
        return outputs - self.nodes[self.num_layers-1];

    def backprop(self, layer, delta, previous_weights=None):
        # calc deltas for the first layer
        # update the weights before the output layers
        # calc deltas farther back

        # if layer is hidden layer right before output layer
        if layer == self.num_layers-2 and previous_weights is None:
            pre_activation_node_layer = self.pre_activation_nodes[layer+1]
            node_layer = self.nodes[layer]
            delta_k = delta*self.sigmoid_derivative(pre_activation_node_layer)
            gradient_descent = np.outer(node_layer, delta_k);
            self.W[layer] += self.learning_rate*gradient_descent
            return delta_k

        weights = self.W[layer];
        pre_activation_node_layer = self.pre_activation_nodes[layer+1]
        node_layer = self.nodes[layer]
        delta_l = delta@previous_weights.transpose() * self.sigmoid_derivative(pre_activation_node_layer);
        gradient_descent = np.outer(node_layer, delta_l)
        self.W[layer] += self.learning_rate*gradient_descent
        return delta_l;

    # inputs and outputs are numpy arrays with correct shape
    def train(self, inputs, outputs, num_iterations):
        if len(inputs) != len(outputs):
            print("Number of inputs different from number of outputs. Returning.")
            return;

        progress_thread = ProgressBarThread(num_iterations)
        progress_thread.start()

        print("-> Training started")
        start_time = time.time()
        for i in range(num_iterations):
            for j in range(len(inputs)):
                self.nodes[0] = inputs[j]
                self.pre_activation_nodes[0] = inputs[j]

                # update nodes[1] through nodes[num_layers-1], inclusive
                for k in range(self.num_layers-1):
                    self.feedforward(k+1)

                delta = self.error(outputs[j]);
                delta = self.backprop(self.num_layers-2, delta, None)
                for k in range(self.num_layers-3, -1, -1):
                    delta = self.backprop(k, delta, self.W[k+1])
            progress_thread.update(i)
        end_time = time.time()
        progress_thread.stop(finished=True)
        progress_thread.join()

        print("\n\n-> Training finished")
        print(f"-> Time elapsed: {end_time-start_time:.8f} seconds")

    def predict_one(self, input):
        self.nodes[0] = input;
        for i in range(self.num_layers-1):
            self.feedforward(i+1)
        return self.nodes[self.num_layers-1]

    def predict_multiple(self, inputs):
        results = []
        for input in inputs:
            results.append(self.predict_one(input))
        return results
