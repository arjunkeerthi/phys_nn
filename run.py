from phys_nn import NN
import numpy as np
from sklearn import datasets
import sklearn.metrics

def run_NN():
    # goal: have input[i] as inputs to the NN give back target[i] as the output
    #input = np.array([[0,0,1],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
    #target = np.array([[0],[1],[0],[1],[1],[0]])

    data = datasets.load_iris()
    x = data["data"]
    x = (x-x.mean())/x.std()
    y = data["target"]
    y = np.eye(3)[y]

    # Took 444.64 seconds
    # test_NN = NN([4,6,10,20,10,6,3],[0,0,0,0,0,0])
    # test_NN.train(x, y, 10000)

    # Took 209.85 seconds
    # test_NN = NN([4,10,7,3],[0,0,0])
    # test_NN.train(x, y, 10000)

    # Took 82.89 seconds for 10000 iterations
    # Took 69.41 seconds with cython
    # Took 814.78 seconds for 100000 iterations
    test_NN = NN([4,5,3],[0,0])
    test_NN.train(x, y, 10000)

    # Took 61.77 seconds, but not very accurate
    # test_NN = NN([4,3],[0])
    # test_NN.train(x, y, 10000)

    print("\n-> Results")
    results = test_NN.predict_multiple(x)
    np.set_printoptions(formatter={'all':lambda x: f'{x:.8f}'})
    for i in range(len(results)):
         print(str(results[i]))

if __name__ == "__main__":
    run_NN()
