import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from mlp import MLP, Layer, InputLayer, OutputLayer, MultinoulliML
from utils import Adam


def epoch_it(X, epochs, batchsize):
    n, _ = X.shape
    N = int(n / batchsize)
    for _ in range(epochs):
        for i in range(N):
            yield X[i * batchsize : (i + 1) * batchsize]


def file_it(path, skiplines=0):
    with open(path) as f:
        for _ in range(skiplines):
            f.readline()
        yield from f


def train():
    class DropoutConfig:
        enabled = False
        p_hidden = 0.7
        p_input = 0.8

    network = MLP(
        [
            InputLayer(28 * 28),
            Layer(5),
            Layer(5),
            Layer(5),
            OutputLayer(10),
        ],
        model=MultinoulliML(),
        optimizer=Adam(eps=0.1, clip_threshold=None),
        dropout=DropoutConfig,
    )

    batchsize = 20
    epochs = 1

    data = np.loadtxt("../data/train.csv", skiprows=1, delimiter=",", dtype=int)
    valset = data[:100]
    valset = (valset[:, 1:] / 255, valset[:, 0])
    trainset = data[100:]
    it = epoch_it(trainset, epochs, batchsize=batchsize)

    network.init_training(batchsize=batchsize, validation_set=valset)

    for batch in it:
        X, Y = (batch[:, 1:] / 255, batch[:, 0])
        cost, accuracy = network.train_minibatch((X, Y))
        print(accuracy, cost)
    network.cleanup_training()

    return network
