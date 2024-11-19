import numpy as np
from mlp import MLP, Layer, InputLayer, OutputLayer, MultinoulliML
from utils import Adam
from visualization import MLPVisualization


class MLPInterface:
    def __init__(self, mlp, training_it):
        self._mlp = mlp
        self._training_it = training_it
        self.layers = [LayerInterface(layer) for layer in self._mlp.layers]

        self.hooks = {
            "train_batch": self.train_batch,
            "stop_training": self.stop_training,
        }

    def train_batch(self):
        try:
            self._mlp.train_minibatch(next(self._training_it))
            return True
        except StopIteration:
            return False

    def stop_training(self):
        self._mlp.cleanup_training()


class LayerInterface:
    def __init__(self, layer):
        self._layer = layer
        self.width = self._layer.width

    @property
    def weights(self):
        if self._layer.dropout.enabled:
            return self._layer.W_s.T
        else:
            return self._layer.W.T

    @property
    def activations(self):
        A = self._layer.A
        if len(A.shape) > 1:
            return A[0]
        else:
            return A


class epoch_it:
    def __init__(self, X, epochs, batchsize):
        self.X = X
        self.epochs = epochs
        self.batchsize = batchsize
        n, _ = self.X.shape
        self.i = 0
        self.m = int(n / self.batchsize)
        self.N = self.epochs * self.m

    def __iter__(self):
        return self

    def __next__(self):
        if self.epochs > 0:
            if self.i < self.m:
                batch = self.X[self.i * self.batchsize : (self.i + 1) * self.batchsize]
                self.i += 1
                return batch
            else:
                self.i = 0
                self.epochs -= 1
                return self.__next__()
        else:
            raise StopIteration


def main():
    class DropoutConfig:
        enabled = False
        p_hidden = 0.7
        p_input = 0.8

    network = MLP(
        [
            InputLayer(28 * 28),
            Layer(100),
            Layer(100),
            Layer(100),
            Layer(100),
            OutputLayer(10),
        ],
        model=MultinoulliML(),
        optimizer=Adam(eps=0.1, clip_threshold=None),
        dropout=DropoutConfig,
    )

    batchsize = 20
    epochs = 1

    data = np.loadtxt("data/train.csv", skiprows=1, delimiter=",", dtype=int)
    valset = data[:100]
    valset = (valset[:, 1:] / 255, valset[:, 0])
    trainset = data[100:]
    it = map(
        lambda batch: (batch[:, 1:] / 255, batch[:, 0]),
        epoch_it(trainset, epochs, batchsize=batchsize),
    )

    network.init_training(batchsize=batchsize, validation_set=valset)

    mlp_interface = MLPInterface(network, training_it=it)

    MLPVisualization(mlp_interface, dx=10, dy=3, r=1)


if __name__ == "__main__":
    main()
