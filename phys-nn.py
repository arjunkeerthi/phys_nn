import numpy as np
import random

class NN:
    # num_layers = number of layers in network
    # layer_node_count = list of number of nodes in each layers
    # - First element: Number of nodes in input layer, Last: output layer
    # bias = Bias for the corresponding layer counted in layer_node_count
    def __init__(self, num_layers=3, layer_node_count=[2,2,2], bias=[1.0, 1.0]):
        self.learning_rate = 0.1
        self.num_iterations = 10000

        # define the activation function using python's ability to treat functions as objects
        self.activ_func = np.tanh
        self.num_layers = num_layers

        # instantiate the weights matrices
        self.W = []
        for i in range(num_layers-1):
            # Add weights between i and i+1 layer
            weights = np.ndarray(shape=(layer_node_count[i], layer_node_count[i+1])))
            W.add(weights)

        self.bias = bias

        # nodes[i] is a vector of length layer_node_count[i]
        self.nodes = []
        for i in range(num_layers):
            self.nodes[i] = np.zeros(layer_node_count[i])

    # give the layer value to update via x_j = s(w_ij x_i + b_j)
    def feedforward(layer):
        if layer == 0:
            print("can not feedforward to the input layer")
            return
        if layer >= num_layers:
            print("feedforward out of bounds")
            return

        # iterate over the nodes in the layer we're updating
        for j in range(len(nodes[layer])):
            layer_W = W[layer-1]
            signal = 0
            # use the value of the nodes in the previous position
            for i in range(len(nodes[layer-1])):
                signal+= layer_W[i][j] * nodes[layer-1][i] + bias[layer-1]

                # Test code for later
                #col = layer_W[:,i]
                #input_layer = np.transpose()

            nodes[layer][j] = activ_func(signal)

    # x_k = s(sum(w_jk * s(sum(w_ij * x_i) + b_j)) + b_k)
    def backprop():

    # inputs and outputs are numpy arrays with correct shape
    def train(inputs, outputs):
        self.nodes[0] = inputs
        self.nodes[len(self.nodes)-1] = outputs
        for i in range(num_layers-1):
            self.feedforward(self.nodes[i+1])


# goal: have input[i] as inputs to the NN give back target[i] as the output
input = np.array([[0,0,1],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
target = np.array([[0],[1],[0],[1],[1],[0]])

print(input)
print(target)



# num_layers = number of layers in network
# layer_node_count = number of nodes in each layer
def forward(num_layers, layer_node_count):

    # instantiate the biases

    # vectors for the nodes at each layer

    #

# this is this x_j = s(w_ij x_i + b_j)
def feedforward(weights, nodes, bias):
    output_nodes = np.array
    for i in range(len(nodes)):
