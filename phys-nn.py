import numpy as np
import copy

class NN:
    # num_layers = number of layers in network
    # layer_node_count = list of number of nodes in each layers
    # - First element: Number of nodes in input layer, Last: output layer
    # bias = Bias for the corresponding layer counted in layer_node_count
    def __init__(self, num_layers=3, layer_node_count=[2,2,2], bias=[1.0, 1.0]):
        self.learning_rate = 0.05
        self.num_iterations = 100

        # define the activation function using python's ability to treat functions as objects
        self.activ_func = np.tanh
        self.activ_func_deriv = lambda x : 1 - (np.tanh(x))**2

        self.num_layers = num_layers

        # instantiate the weights matrices
        self.W = []
        for i in range(num_layers-1):
            # Add weights between i and i+1 layer
            #weights = np.ndarray(shape=(layer_node_count[i], layer_node_count[i+1]))
            weights = np.random.rand(layer_node_count[i], layer_node_count[i+1])*2-1
            self.W.append(weights)

        # Should be length of num_layers-1
        self.bias = bias

        # nodes[i] is a vector of length layer_node_count[i]
        self.nodes = []
        for i in range(num_layers):
            layer = np.zeros(layer_node_count[i])
            self.nodes.append(layer);

    # give the layer value to update via x_j = s(w_ij x_i + b_j)
    def feedforward(self, layer): # finished
        if layer == 0:
            print("can not feedforward to the input layer")
            return
        if layer >= self.num_layers:
            print("feedforward out of bounds")
            return

        # iterate over the nodes in the layer we're updating
        # for j in range(len(self.nodes[layer])):
        #     layer_W = self.W[layer-1]
        #     signal = 0
        #     # use the value of the nodes in the previous position
        #     # sum(w_ij x_i + b_j)
        #     for i in range(len(self.nodes[layer-1])):
        #         signal += layer_W[i][j] * self.nodes[layer-1][i] + self.bias[layer-1]
        #
        #     # pass the signal through the activation function and update the nodes
        #     self.nodes[layer][j] = self.activ_func(signal)

        # Test code for later
        weights = self.W[layer-1]
        prev_node_layer = self.nodes[layer-1]
        self.nodes[layer] = self.activ_func(prev_node_layer@weights);

    # Calculate error in output layer
    def error(self, outputs):
        return outputs - self.nodes[self.num_layers-1];

    def backprop(self, layer, delta):
        # calc deltas for the first layer
        # update the weights before the output layers
        # calc deltas farther back
        #for i in range(len(nodes[layer])):
        #    layer_W = self.W[layer].transpose();
        weights = self.W[layer];
        node_layer = self.nodes[layer]
        delta_l = weights@delta * self.activ_func_deriv(node_layer);
        gradient_descent = self.nodes[layer]*delta_l;
        self.W[layer] = self.W[layer] - self.learning_rate*gradient_descent.reshape(-1,1);
        return delta_l;


    # inputs and outputs are numpy arrays with correct shape
    def train(self, inputs, outputs, num_iterations):
        if len(inputs) != len(outputs):
            print("Number of inputs different from number of outputs. Returning.")
            return;

        print("\n***STARTING TRAINING***")
        initial_weights = copy.deepcopy(self.W)
        for i in range(num_iterations):
            print(f"***ITERATION {i+1}***")
            for j in range(len(inputs)):
                print("\n" + "-"*90 + "\n")
                print(f"Input #{i+1}:");
                self.nodes[0] = inputs[j];
                # print("\nFeedforward stage:")
                # print("Nodes before:")
                # print(self.nodes);
                for i in range(self.num_layers-1):
                    # update nodes[1] through nodes[num_layers-1], inclusive
                    self.feedforward(i+1)
                # print("Nodes after:")
                # print(self.nodes)
                # print("\nBackprop step:")
                # calculate the error
                delta_L = self.error(outputs[j]);
                curr_layer = self.num_layers-2;
                print("Error:")
                print(delta_L)
                # print("W before:")
                # print(self.W)
                delta_l = self.backprop(curr_layer, delta_L);
                for i in range(self.num_layers-3, -1, -1):
                    delta_l = self.backprop(i, delta_l);
                # print("W after:")
                # print(self.W);
            print("\n" + "-"*90 + "\n")

        #print("\n" + "-"*90 + "\n")
        print("Weight comparison:")
        print("Initial:")
        print(initial_weights)
        print("Final:")
        print(test_NN.W)

if __name__ == "__main__":
    # goal: have input[i] as inputs to the NN give back target[i] as the output
    input = np.array([[0,0,1],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
    target = np.array([[0],[1],[0],[1],[1],[0]])

    test_NN = NN(3,[3,3,1],[1.0,1.0]);
    #initial_weights = copy.deepcopy(test_NN.W)
    test_NN.train(input, target, 50)

    # print("\n***STARTING TRAINING***")
    # for j in range(100):
    #     for i in range(len(input)):
    #         print("\n" + "-"*90 + "\n")
    #         print(f"Iteration {i+1}:");
    #         test_NN.train(input[i], target[i]);

    # print("\n" + "-"*90 + "\n")
    # print("Weight comparison:")
    # print("Initial:")
    # print(initial_weights)
    # print("Final:")
    # print(test_NN.W)
