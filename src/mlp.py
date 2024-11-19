import numpy as np
from utils import Activation, ReLU, Softmax, Adam


class LayerBase:
    rng = None
    dropout = None

    def __init__(self, width):
        self.width = width
        self.parent = None
        self.child = None

    def prepend(self, layer):
        self.parent = layer
        self.parent.child = self

    def feedforward(self):
        pass

    def backprop(self):
        pass


class LayerDroppable(LayerBase):
    def __init__(self, width):
        super().__init__(width)
        self.p_dropout = None
        self.feedforward = self._feedforward

    def _feedforward(self):
        pass

    def _feedforward_dropout(self):
        pass

    def init_training(self, batchsize):
        if self.dropout.enabled:
            self.init_dropout(batchsize)

    def cleanup_training(self):
        if self.dropout.enabled:
            self.cleanup_dropout()

    def init_dropout(self, batchsize):
        self.dropout_mask = lambda: self.rng.binomial(
            n=1, p=self.p_dropout, size=(batchsize, self.width)
        )
        self.feedforward = self._feedforward_dropout
        self._ff_toggle = self._feedforward

    def cleanup_dropout(self):
        self.toggle_dropout_off()

    def toggle_dropout_off(self):
        np.multiply(self.child.W, self.p_dropout, out=self.child.W_s)
        np.multiply(self.child.b, self.p_dropout, out=self.child.b_s)
        self.toggle_dropout()

    def toggle_dropout(self):
        (self.child.W, self.child.W_s) = (self.child.W_s, self.child.W)
        (self.child.b, self.child.b_s) = (self.child.b_s, self.child.b)
        (self.feedforward, self._ff_toggle) = (self._ff_toggle, self.feedforward)


class Layer(LayerDroppable):
    def __init__(self, width, activation: Activation = ReLU):
        super().__init__(width)
        self.activation = activation

        self.batchsize = None
        self.Z = None
        self.A = None
        self.dsdZ = None
        self.dJdZ = None
        self.dJdWn = None
        self.dJdW = None
        self.dJdbn = None
        self.dJdb = None
        self.dZdA = None
        self.temp = None

        self.W = None
        self.b = None

        self.A_ = None
        self.Z_ = None
        self.temp_ = None

        self.p_dropout = None
        self.W_s = None
        self.b_s = None

    def prepend(self, layer):
        super().prepend(layer)
        self.init_params()

    def init_params(self):
        n, m = (self.parent.width, self.width)
        a = np.sqrt(6 / (m + n))
        self.W = self.rng.uniform(low=-a, high=a, size=(n, m))
        self.b = np.zeros(m)

    def init_training(self, batchsize):
        super().init_training(batchsize)
        self.batchsize = batchsize
        self.Z = np.empty((batchsize, self.width))
        self.A = np.empty_like(self.Z)
        self.dsdZ = self.Z
        self.dJdZ = np.empty_like(self.Z)
        self.dJdWn = np.empty((batchsize, *self.W.shape))
        self.dJdW = np.empty_like(self.W)
        self.dJdbn = self.dJdZ
        self.dJdb = np.empty(self.width)
        self.dZdA = self.W.T
        self.temp = np.empty(batchsize)

    def init_feedforward(self, size):
        self.A_ = np.empty((size, self.width))
        self.Z_ = np.empty_like(self.A_)
        self.temp_ = np.empty(size)

    def cleanup_feedforward(self):
        self.A_ = None
        self.Z_ = None
        self.temp_ = None

    def toggle_feedforward(self):
        (self.A, self.A_) = (self.A_, self.A)
        (self.Z, self.Z_) = (self.Z_, self.Z)
        (self.temp, self.temp_) = (self.temp_, self.temp)

    def init_dropout(self, batchsize):
        super().init_dropout(batchsize)
        self.p_dropout = self.dropout.p_hidden
        self.W_s = np.copy(self.W)
        self.b_s = np.empty_like(self.b)

    def cleanup_training(self):
        super().cleanup_training()

        self.batchsize = None
        self.Z = None
        self.A = None
        self.dsdZ = None
        self.dJdZ = None
        self.dJdWn = None
        self.dJdW = None
        self.dJdbn = None
        self.dJdb = None
        self.dZdA = None
        self.temp = None

    def cleanup_dropout(self):
        super().cleanup_dropout()

        self.W_s = None
        self.b_s = None
        self.p_dropout = None

    def _feedforward(self):
        np.matmul(self.parent.A, self.W, out=self.Z)
        np.add(self.Z, self.b, out=self.Z)
        self.activation._func(self.Z, out=self.A, temp=self.temp)

    def _feedforward_dropout(self):
        self._feedforward()
        np.multiply(self.A, self.dropout_mask(), out=self.A)

    def _compute_dJdZ(self):
        self.activation.deriv(self.Z, self.A, out=self.dsdZ, temp=self.temp)
        np.matmul(self.child.dJdZ, self.child.dZdA, out=self.dJdZ)
        np.multiply(self.dsdZ, self.dJdZ, out=self.dJdZ)

    def backprop(self):
        self._compute_dJdZ()
        np.multiply(
            self.parent.A[:, :, np.newaxis], self.dJdZ[:, np.newaxis, :], out=self.dJdWn
        )
        np.sum(self.dJdWn, axis=0, out=self.dJdW)
        np.divide(self.dJdW, self.batchsize, out=self.dJdW)
        np.sum(self.dJdbn, axis=0, out=self.dJdb)
        np.divide(self.dJdb, self.batchsize, out=self.dJdb)


class InputLayer(LayerDroppable):
    def __init__(self, width):
        super().__init__(width)
        self.X = None
        self.A = None
        self.A_ = None

    def init_training(self, batchsize):
        super().init_training(batchsize)
        self.A = np.empty((batchsize, self.width))

    def init_dropout(self, batchsize):
        super().init_dropout(batchsize)
        self.p_dropout = self.dropout.p_input

    def cleanup_training(self):
        super().cleanup_training()
        self.A = None

    def init_feedforward(self, size):
        self.A_ = np.empty((size, self.width))

    def cleanup_feedforward(self):
        self.A_ = None

    def toggle_feedforward(self):
        (self.A, self.A_) = (self.A_, self.A)

    def _feedforward(self):
        np.copyto(src=self.X, dst=self.A)

    def _feedforward_dropout(self):
        np.multiply(
            self.X,
            self.dropout_mask(),
            out=self.A,
        )


class OutputLayer(Layer):
    def __init__(self, width):
        super().__init__(width, activation=None)

    def config(self, activation, compute_dJdZ):
        self.activation = activation
        self._compute_dJdZ = lambda: compute_dJdZ(self)

    def init_dropout(self, batchsize):
        self.W_s = np.copy(self.W)
        self.b_s = np.empty_like(self.b)

    def toggle_dropout(self):
        pass

    def toggle_dropout_off(self):
        pass

    def cleanup_dropout(self):
        self.W_s = None
        self.b_s = None


class DropoutConfigDefault:
    enabled = True
    p_hidden = 0.5
    p_input = 0.8


class MLP:
    def __init__(self, layers, model, optimizer=Adam(), dropout=DropoutConfigDefault):
        self.dropout = dropout
        LayerBase.rng = np.random.default_rng(seed=12345)
        LayerDroppable.dropout = dropout

        self.layers = layers
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.model = model
        self.optimizer = optimizer

        self.output_layer.config(
            activation=self.model.activation, compute_dJdZ=self.model.compute_dJdZ
        )

        child = self.output_layer
        for parent in reversed(self.layers[:-1]):
            child.prepend(parent)
            child = parent

        for layer in self.layers:
            # initialize layers to accept single examples
            layer.init_feedforward(1)
            layer.toggle_feedforward()

    def init_training(self, batchsize, validation_set=None):
        self.validation_set = validation_set

        for layer in self.layers:
            # initialize layers to accept batches
            layer.toggle_feedforward()
            layer.init_training(batchsize)

        # initialize layers to accept inputs of size of
        # validation set on their alternative buffers
        if self.validation_set is not None:
            X, Y = self.validation_set
            N, _ = X.shape
            for layer in self.layers:
                layer.init_feedforward(N)

        self.model.init_training(batchsize, validation_set, self.output_layer)
        self.optimizer.init_training(self.layers[1:])

    def cleanup_training(self):
        for layer in self.layers:
            layer.cleanup_training()
            layer.cleanup_feedforward()
            # initialize layers to accept single examples again
            layer.init_feedforward(1)
            layer.toggle_feedforward()

    def train_minibatch(self, minibatch):
        X, Y = minibatch
        self._feedforward(X)
        self.model.outputs = (self.output_layer.Z, self.output_layer.A, Y)

        layer = self.output_layer
        while layer:
            layer.backprop()
            layer = layer.parent

        self.optimizer.update_params()

        if self.validation_set is not None:
            X, Y = self.validation_set

            # switch to alternative buffers
            for layer in self.layers:
                layer.toggle_feedforward()

            if self.dropout.enabled:
                for layer in self.layers:
                    layer.toggle_dropout_off()

            self._feedforward(X)

            self.model.outputs = (self.output_layer.Z, self.output_layer.A, Y)

            if self.dropout.enabled:
                for layer in self.layers:
                    layer.toggle_dropout()

            # switch back to main buffers
            for layer in self.layers:
                layer.toggle_feedforward()

            return (self.model.loss, self.model.accuracy)

    def _feedforward(self, X):
        layer = self.input_layer
        layer.X = X

        while layer:
            layer.feedforward()
            layer = layer.child

    def feedforward(self, x):
        layer = self.input_layer
        layer.A = x
        layer = layer.child
        while layer:
            layer.A = layer.activation.func(layer.parent.A @ layer.W + layer.b)
            layer = layer.child
        return np.argmax(self.output_layer.A)


class MultinoulliML:
    activation = Softmax

    def __init__(self):
        self.outputs = (None, None)

    def init_training(self, batchsize, validation_set, output_layer):
        self.batchsize = batchsize
        self.index = np.arange(batchsize)
        if validation_set is not None:
            X, _ = validation_set
            self.valsetsize, _ = X.shape

            self.index_ = np.arange(self.valsetsize)
            self.loss_ = np.empty(self.valsetsize)
            self.temp1_ = np.empty_like(self.loss_)
            self.temp2_ = np.empty(self.valsetsize, dtype=int)

            self.temp_ = np.empty((self.valsetsize, output_layer.width))

    def compute_dJdZ(self, output_layer):
        _, Y_, Y = self.outputs
        np.copyto(src=Y_, dst=output_layer.dJdZ)
        output_layer.dJdZ[self.index, Y] -= 1

    @property
    def loss(self):
        Z, _, Y = self.outputs
        np.max(Z, axis=-1, out=self.loss_)
        np.subtract(Z, self.loss_[:, np.newaxis], out=self.temp_)
        np.exp(self.temp_, out=self.temp_)
        np.sum(self.temp_, axis=-1, out=self.temp1_)
        np.log(self.temp1_, out=self.temp1_)
        np.subtract(self.loss_, Z[self.index_, Y], out=self.loss_)
        np.add(self.loss_, self.temp1_, out=self.loss_)
        return np.sum(self.loss_) / self.valsetsize

    @property
    def accuracy(self):
        _, Y_, Y = self.outputs
        np.argmax(Y_, axis=-1, out=self.temp2_)
        np.equal(self.temp2_, Y, out=self.temp2_)
        return np.average(self.temp2_)
