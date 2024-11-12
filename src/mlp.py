import numpy as np


class Activation:
    @classmethod
    def func(_, Z): ...

    @classmethod
    def _func(_, Z, out, temp): ...

    @classmethod
    def deriv(_, Z, A, out, temp): ...


class ReLU(Activation):
    @classmethod
    def func(_, Z):
        return np.where(Z > 0, Z, 0)

    @classmethod
    def _func(_, Z, out, temp):
        np.multiply(Z, Z > 0, out=out)

    @classmethod
    def deriv(_, Z, A, out, temp):
        np.copyto(src=A, dst=out)
        np.putmask(out, out > 0, 1)


class Softmax(Activation):

    @classmethod
    def func(_, Z):
        z = np.exp(Z - Z.max())
        return z / np.sum(z)

    @classmethod
    def _func(_, Z, out, temp):
        np.max(Z, axis=-1, out=temp)
        np.subtract(Z, temp[:, np.newaxis], out=out)
        np.exp(out, out=out)
        np.sum(out, axis=-1, out=temp)
        np.divide(out, temp[:, np.newaxis], out=out)

    @classmethod
    def deriv(_, Z, A, out, temp):
        np.square(A, out=out)
        np.multiply(out, -1, out=out)
        np.add(out, A, out=out)


class LayerBase:
    def __init__(self, width):
        self.width = width
        self.parent = None
        self.child = None

    def append(self, child):
        self.child = child
        self.child.parent = self
        n, m = (self.width, self.child.width)
        a = np.sqrt(6 / (m + n))
        self.child.W = self.rng.uniform(low=-a, high=a, size=(n, m))

    def feedforward(self):
        pass

    def backprop(self):
        pass


class InputLayer(LayerBase):
    def __init__(self, width):
        super().__init__(width)


class Layer(LayerBase):
    def __init__(self, width, activation: Activation = ReLU):
        super().__init__(width)
        self.activation = activation
        self.b = np.zeros(self.width)

    def init_training(self, batchsize):
        self.Z = np.empty((batchsize, self.width))
        self.A = np.empty_like(self.Z)
        self.dsdZ = self.Z
        self.dJdZ = np.empty_like(self.Z)
        self.dJdWn = np.empty((batchsize, *self.W.shape))
        self.dJdW = np.empty_like(self.W)
        self.dJdbn = self.dJdZ
        self.dJdb = np.empty(self.width)
        self.temp = np.empty(batchsize)
        self.batchsize = batchsize

    def feedforward(self):
        np.matmul(self.parent.A, self.W, out=self.Z)
        np.add(self.Z, self.b, out=self.Z)
        self.activation._func(self.Z, out=self.A, temp=self.temp)

    def _compute_dJdZ(self):
        self.activation.deriv(self.Z, self.A, out=self.dsdZ, temp=self.temp)
        np.matmul(self.child.dJdZ, self.child.W.T, out=self.dJdZ)
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


class OutputLayer(Layer):
    def __init__(self, width):
        super().__init__(width, activation=None)

    def config(self, activation, compute_dJdZ):
        self.activation = activation
        self.__compute_dJdZ = compute_dJdZ

    def _compute_dJdZ(self):
        self.__compute_dJdZ(self)


class Adam:
    def __init__(self, eps=0.001, r1=0.9, r2=0.999, d=1e-8):
        self.eps = eps
        self.r1 = r1
        self.r2 = r2
        self.d = d

    def init_training(self, layers):
        self.t = 0
        self.data = []
        for layer in layers:
            sW = np.zeros_like(layer.W)
            sW_ = np.empty_like(layer.W)
            rW = np.zeros_like(layer.W)
            rW_ = np.empty_like(layer.W)

            sb = np.zeros_like(layer.b)
            sb_ = np.empty_like(layer.b)
            rb = np.zeros_like(layer.b)
            rb_ = np.empty_like(layer.b)
            self.data.append((layer, (sW, sW_, rW, rW_), (sb, sb_, rb, rb_)))

    def update_params(self):
        self.t += 1
        r1, r2, t = (self.r1, self.r2, self.t)
        for layer, W_, b_ in self.data:
            for (s, s_, r, r_), g, p in (
                (W_, layer.dJdW, layer.W),
                (b_, layer.dJdb, layer.b),
            ):
                np.multiply(r1, s, out=s)
                np.multiply(1 - r1, g, out=s_)
                np.add(s, s_, out=s)

                np.multiply(r2, r, out=r)
                np.square(g, out=r_)
                np.multiply(1 - r2, r_, out=r_)
                np.add(r, r_, out=r)

                np.divide(s, 1 - np.power(r1, t), out=s)
                np.divide(r, 1 - np.power(r2, t), out=r)

                np.sqrt(r, out=r)
                np.add(r, self.d, out=r)
                np.multiply(s, -self.eps, out=s)
                np.divide(s, r, out=s)

                np.add(p, s, out=p)


class MLP:
    def __init__(self, layers, model, optimizer=Adam()):
        LayerBase.rng = np.random.default_rng(seed=12345)

        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.layers = layers[1:]
        self.model = model
        self.optimizer = optimizer

        self.output_layer.config(
            activation=self.model.activation, compute_dJdZ=self.model.compute_dJdZ
        )

        parent = self.input_layer
        for layer in self.layers:
            parent.append(layer)
            parent = parent.child

    def init_training(self, batchsize, validation_set=None):
        self.validation_set = validation_set

        for layer in self.layers:
            layer.init_training(batchsize)

        self.model.init_training(batchsize, validation_set)

        self.optimizer.init_training(self.layers)

    def train_minibatch(self, minibatch):
        X, Y = minibatch
        layer = self.input_layer
        layer.A = X

        while layer:
            layer.feedforward()
            layer = layer.child

        self.model.outputs = (self.output_layer.A, Y)

        layer = self.output_layer
        while layer:
            layer.backprop()
            layer = layer.parent

        self.optimizer.update_params()

        if self.validation_set is not None:
            X, Y = self.validation_set

            layer = self.input_layer
            layer.A = X

            while layer:
                layer.feedforward()
                layer = layer.child

            self.model.outputs = (self.output_layer.A, Y)
            print(self.model.loss)

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

    def init_training(self, batchsize, validation_set):
        self.batchsize = batchsize
        self.index = np.arange(batchsize)
        if validation_set is not None:
            X, _ = validation_set
            m, _ = X.shape
            self.loss_ = np.empty(m)
            self.index_ = np.arange(m)

    def compute_dJdZ(self, output_layer):
        Y_, Y = self.outputs
        np.copyto(src=Y_, dst=output_layer.dJdZ)
        output_layer.dJdZ[self.index, Y] -= 1

    @property
    def loss(self):
        Y_, Y = self.outputs
        np.log(Y_[self.index_, Y], out=self.loss_)
        return -np.sum(self.loss_)
