import numpy as np
from utils import Activation, ReLU, Softmax, Adam
import os.path


class MLogit:
    class Logloss:
        def init_training(self, output_shape):
            self.setlen = output_shape[-2]
            self.index = np.arange(self.setlen)
            self.temp = np.empty(output_shape[:-1])
            self.temp1 = np.empty_like(self.temp)
            self.temp2 = np.empty(output_shape)

        def cleanup_training(self):
            self.setlen = None
            self.index = None
            self.temp = None
            self.temp1 = None
            self.temp2 = None

        def get_loss(self, Z, Y, out=None):
            np.max(Z, axis=-1, out=self.temp)
            np.subtract(Z, np.reshape(self.temp, (*self.temp.shape, 1)), out=self.temp2)
            np.exp(self.temp2, out=self.temp2)
            np.sum(self.temp2, axis=-1, out=self.temp1)
            np.log(self.temp1, out=self.temp1)
            np.add(self.temp, self.temp1, out=self.temp)
            np.subtract(self.temp, Z[..., self.index, Y], out=self.temp)

            if out is not None:
                np.sum(self.temp, axis=-1, out=out)
                np.divide(out, self.setlen, out=out)
            else:
                return np.sum(self.temp, axis=-1) / self.setlen

        def compute_dJdZ(self, Y_, Y, out):
            np.copyto(src=Y_, dst=out)
            out[self.index, Y] -= 1

    class Accuracy:
        def init_training(self, output_shape):
            self.temp = np.empty(output_shape[:-1], dtype=int)

        def cleanup_training(self):
            self.temp = None

        def get_accuracy(self, Y_, Y):
            np.argmax(Y_, axis=-1, out=self.temp)
            np.equal(Y, self.temp, out=self.temp)
            return np.average(self.temp)

    activation = Softmax

    def __init__(self, output_layer):
        self.output_layer = output_layer
        self.train_loss = self.Logloss()
        self.train_accuracy = self.Accuracy()
        self.val_loss = self.Logloss()
        self.val_accuracy = self.Accuracy()

    def init_training(self, minibatch_output, valset_output):
        self.train_loss.init_training(minibatch_output)
        self.train_accuracy.init_training(minibatch_output)
        self.val_loss.init_training(valset_output)
        self.val_accuracy.init_training(valset_output)

    def cleanup_training(self):
        self.train_loss.cleanup_training()
        self.train_accuracy.cleanup_training()
        self.val_loss.cleanup_training()
        self.val_accuracy.cleanup_training()

    def compute_dJdZ(self):
        self.train_loss.compute_dJdZ(
            Y_=self.output_layer.A, Y=self.output_layer.Y, out=self.output_layer.dJdZ
        )

    @property
    def training_loss(self):
        return self.train_loss.get_loss(Y=self.output_layer.Y, Z=self.output_layer.Z)

    @property
    def training_accuracy(self):
        return self.train_accuracy.get_accuracy(
            Y_=self.output_layer.A, Y=self.output_layer.Y
        )

    @property
    def valset_loss(self):
        return self.val_loss.get_loss(Y=self.output_layer.Y, Z=self.output_layer.Z)

    @property
    def valset_accuracy(self):
        return self.val_accuracy.get_accuracy(
            Y_=self.output_layer.A, Y=self.output_layer.Y
        )


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
        self._ff_toggle = self._feedforward_dropout

    def _feedforward(self):
        pass

    def _feedforward_dropout(self):
        pass

    def init_dropout(self, batchsize):
        self.dropout_mask = lambda: self.rng.binomial(
            n=1, p=self.p_dropout, size=(batchsize, self.width)
        )

    def rescale_weights_dropout(self):
        np.multiply(self.child.W, self.p_dropout, out=self.child.W_s)

    def toggle_dropout(self):
        (self.child.W, self.child.W_s) = (self.child.W_s, self.child.W)
        (self.feedforward, self._ff_toggle) = (self._ff_toggle, self.feedforward)

    def cleanup_dropout(self):
        self.dropout_mask = None


class Layer(LayerDroppable):
    def __init__(self, width, activation: Activation = ReLU):
        super().__init__(width)
        self.activation = activation

        self.batchsize = None
        self.dsdZ = None
        self.dJdZ = None
        self.dJdWn = None
        self.dJdW = None
        self.dJdbn = None
        self.dJdb = None
        self.dZdA = None

        self.W = None
        self.b = None

        self.A = None
        self.Z = None
        self.temp = None

        self.A_ = None
        self.Z_ = None
        self.temp_ = None

        self.p_dropout = None
        self.W_s = None
        self.b_s = None

    def init_params(self):
        n, m = (self.parent.width, self.width)
        a = np.sqrt(6 / (m + n))
        self.W = self.rng.uniform(low=-a, high=a, size=(n, m))
        self.b = np.zeros(m)

    def prepend(self, layer):
        super().prepend(layer)
        self.init_params()

    def init_feedforward(self, size):
        self.A_ = np.empty((size, self.width))
        self.Z_ = np.empty_like(self.A_)
        self.temp_ = np.empty(size)

    def toggle_feedforward(self):
        (self.A, self.A_) = (self.A_, self.A)
        (self.Z, self.Z_) = (self.Z_, self.Z)
        (self.temp, self.temp_) = (self.temp_, self.temp)

    def cleanup_feedforward(self):
        self.A_ = None
        self.Z_ = None
        self.temp_ = None

    def init_training(self, batchsize):
        self.batchsize = batchsize

        # init A,Z,temp
        self.init_feedforward(self.batchsize)
        self.toggle_feedforward()

        self.dsdZ = np.empty_like(self.Z)
        self.dJdZ = np.empty_like(self.Z)
        self.dJdWn = np.empty((batchsize, *self.W.shape))
        self.dJdW = np.empty_like(self.W)
        self.dJdbn = self.dJdZ
        self.dJdb = np.empty(self.width)
        self.dZdA = self.W.T

        self.W_opt = np.empty_like(self.W)
        self.b_opt = np.empty_like(self.b)

    def cleanup_training(self):
        self.batchsize = None

        # free all A,Z,temp
        self.toggle_feedforward()
        self.cleanup_feedforward()
        self.toggle_feedforward()
        self.cleanup_feedforward()

        self.dsdZ = None
        self.dJdZ = None
        self.dJdWn = None
        self.dJdW = None
        self.dJdbn = None
        self.dJdb = None
        self.dZdA = None

        (self.W, self.W_opt) = (self.W_opt, self.W)
        (self.b, self.b_opt) = (self.b_opt, self.b)

        self.W_opt = None
        self.b_opt = None

    def init_dropout(self, batchsize):
        super().init_dropout(batchsize)
        self.p_dropout = self.dropout.p_hidden
        self.W_s = np.copy(self.W)

    def cleanup_dropout(self):
        super().cleanup_dropout()
        self.W_s = None
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
            self.parent.A[..., np.newaxis], self.dJdZ[:, np.newaxis, :], out=self.dJdWn
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

    def init_feedforward(self, size):
        self.A_ = np.empty((size, self.width))

    def toggle_feedforward(self):
        (self.A, self.A_) = (self.A_, self.A)

    def cleanup_feedforward(self):
        self.A_ = None

    def init_training(self, batchsize):
        self.init_feedforward(batchsize)
        self.toggle_feedforward()

    def cleanup_training(self):
        self.toggle_feedforward()
        self.cleanup_feedforward()
        self.toggle_feedforward()
        self.cleanup_feedforward()

    def init_dropout(self, batchsize):
        super().init_dropout(batchsize)
        self.p_dropout = self.dropout.p_input

    def cleanup_dropout(self):
        super().cleanup_dropout()
        self.p_dropout = None

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
        self._compute_dJdZ = compute_dJdZ

    def init_dropout(self, _):
        self.W_s = np.copy(self.W)

    def rescale_weights_dropout(self):
        pass

    def toggle_dropout(self):
        pass

    def cleanup_dropout(self):
        self.W_s = None


class DropoutConfigDefault:
    enabled = True
    p_hidden = 0.5
    p_input = 0.8


class MLP:
    def __init__(
        self, layers, model=MLogit, optimizer=Adam(), dropout=DropoutConfigDefault
    ):
        self.dropout = dropout
        LayerBase.rng = np.random.default_rng(seed=12345)
        LayerDroppable.dropout = dropout

        self.layers = layers
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

        self.model = model(self.output_layer)
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

    def init_training(self, batchsize, validation_set):
        self.validation_set = validation_set

        for layer in self.layers:
            # initialize layers to accept batches
            layer.init_training(batchsize)

        # initialize layers to accept inputs of size of
        # validation set on their alternative arrays
        X, _ = self.validation_set
        valsetsize, _ = X.shape
        for layer in self.layers:
            layer.init_feedforward(valsetsize)

        self.model.init_training(
            minibatch_output=(batchsize, self.output_layer.width),
            valset_output=(valsetsize, self.output_layer.width),
        )

        self.optimizer.init_training(self.layers[1:])

        # initialize dropout
        if self.dropout.enabled:
            for layer in self.layers:
                layer.init_dropout(batchsize)

        self.min_valset_loss = np.inf

    def cleanup_training(self):
        for layer in self.layers:
            layer.cleanup_training()
            # initialize layers to accept single examples again
            layer.init_feedforward(1)
            layer.toggle_feedforward()
            layer.cleanup_dropout()

    def train_minibatch(self, minibatch):
        # train
        if self.dropout.enabled:
            for layer in self.layers:
                layer.toggle_dropout()

        X, Y = minibatch
        self._feedforward(X)
        self.output_layer.Y = Y

        training_loss = self.model.training_loss
        training_accuracy = self.model.training_accuracy

        layer = self.output_layer
        while layer:
            layer.backprop()
            layer = layer.parent
        self.optimizer.update_params()

        if self.dropout.enabled:
            for layer in self.layers:
                layer.rescale_weights_dropout()
                layer.toggle_dropout()

        # validate
        X, Y = self.validation_set
        # switch array pointers
        for layer in self.layers:
            layer.toggle_feedforward()

        self._feedforward(X)
        self.output_layer.Y = Y

        validation_loss = self.model.valset_loss
        validation_accuracy = self.model.valset_accuracy

        # switch back array pointers
        for layer in self.layers:
            layer.toggle_feedforward()

        if validation_loss < self.min_valset_loss:
            self.min_valset_loss = validation_loss
            for layer in self.layers[1:]:
                np.copyto(src=layer.W, dst=layer.W_opt)
                np.copyto(src=layer.b, dst=layer.b_opt)

        return (training_loss, training_accuracy, validation_loss, validation_accuracy)

    def save_params(self, path):
        for i, layer in enumerate(self.layers[1:], start=1):
            np.save(os.path.join(path, f"W{i}"), layer.W)
            np.save(os.path.join(path, f"b{i}"), layer.b)

    def load_params(self, path):
        for i, layer in enumerate(self.layers[1:], start=1):
            layer.W = np.load(os.path.join(path, f"W{i}.npy"))
            layer.b = np.load(os.path.join(path, f"b{i}.npy"))

    def _feedforward(self, X):
        layer = self.input_layer
        layer.X = X
        while layer:
            layer.feedforward()
            layer = layer.child

    def feedforward(self, x):
        self._feedforward(x)
        return (np.argmax(self.output_layer.A), self.output_layer.A)


class ParameterAxis:
    def __init__(self, layer_i, type, pos, d, N):
        self.range = None
        self.layer_i = layer_i

        self.d = d
        self.N = N
        self.update_range()

    def update_range(self):
        val = self.value
        self.range = np.linspace(val - self.d, val + self.d, num=self.N)

    @property
    def in_range(self):
        val = self.value
        return self.range.min() < val and val < self.range.max()

    @property
    def value(self):
        return self._get_value()


class LayerGrid:
    def __init__(self, layer, axes):
        self.W_ = np.copy(layer.W)
        self.b_ = np.copy(layer.b)

        self.axes = axes


class ParameterGrid:
    def init_axis(p):
        (layer_i, t, pos, d, N) = p

    def __init__(self, p1, p2, mlp: MLP):
        self.layers = [
            (np.copy(layer.W), np.copy(layer.b), layer.activation)
            for layer in mlp.layers[1:]
        ]
