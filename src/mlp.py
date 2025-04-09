import numpy as np
from optimize import adam


class ff:
    def __init__(self):
        def func(A, W, b):
            np.dot(A, W, out=self.Z_)
            np.add(self.Z_, b, out=self.Z)

        def func_(A, W, b):
            self.Z_ = np.dot(A, W)
            self.Z = np.add(self.Z_, b)
            self.run = func

        self.run = func_


class bprop:
    def __init__(self):

        def func(dJdz_p1, W_p1, A_m1, s_deriv):
            np.einsum("...j,...ij->...i", dJdz_p1, W_p1, out=self.dJdz_, optimize=True)
            np.multiply(s_deriv, self.dJdz_, out=self.dJdz)
            np.multiply(
                A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_
            )
            np.mean(self.dJdW_, axis=0, out=self.dJdW)
            np.mean(self.dJdz, axis=0, out=self.dJdb)

        def func_(dJdz_p1, W_p1, A_m1, s_deriv):
            self.dJdz_ = np.einsum("...j,...ij->...i", dJdz_p1, W_p1, optimize=True)
            self.dJdz = np.multiply(s_deriv, self.dJdz_)
            self.dJdW_ = np.multiply(
                A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :]
            )
            self.dJdW = np.mean(self.dJdW_, axis=0)
            self.dJdb = np.mean(self.dJdz, axis=0)

            self.run = func

        self.run = func_


class relu:
    def __init__(self):
        def func(Z):
            np.greater(Z, 0, out=self.deriv)
            np.multiply(Z, self.deriv, out=self.A)

        def func_(Z):
            self.deriv = np.greater(Z, 0)
            self.A = np.multiply(Z, self.deriv)

            self.run = func

        self.run = func_


class mlogit:
    def __init__(self):

        def func(Z, A_m1, Y=None, loss=False, bprop=False, accuracy=False):
            # compute activations
            np.max(Z, axis=-1, out=self.Z_max_)
            np.subtract(Z, np.expand_dims(self.Z_max_, axis=-1), out=self.A)
            np.exp(self.A, out=self.A)
            np.sum(self.A, axis=-1, out=self.Z_exp_sum_)
            np.divide(self.A, np.expand_dims(self.Z_exp_sum_, axis=-1), out=self.A)

            # compute batch loss
            if loss:
                np.log(self.Z_exp_sum_, out=self.arr)
                np.add(self.arr, self.Z_max_, out=self.arr)
                np.subtract(self.arr, Z[self.index, ..., Y], out=self.arr)
                np.mean(self.arr, axis=0, keepdims=True, out=self.loss)

            if accuracy:
                np.argmax(self.A, axis=-1, out=self.AY)
                np.equal(self.AY, Y, out=self.AY)
                np.mean(self.AY, axis=0, keepdims=True, out=self.accuracy)

            if bprop:
                # compute derivative of per-example loss
                np.copyto(src=self.A, dst=self.dJdz)
                self.dJdz[self.index, ..., Y] -= 1

                # compute derivatives wrt. parameters
                np.multiply(
                    A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_
                )
                np.mean(self.dJdW_, axis=0, out=self.dJdW)
                np.mean(self.dJdz, axis=0, out=self.dJdb)

        def func_(Z, A_m1, Y=None, loss=False, bprop=False, accuracy=False):
            # compute activations
            self.Z_max_ = np.max(Z, axis=-1)
            self.A = np.subtract(Z, np.expand_dims(self.Z_max_, axis=-1))
            np.exp(self.A, out=self.A)
            self.Z_exp_sum_ = np.sum(self.A, axis=-1)
            np.divide(self.A, np.expand_dims(self.Z_exp_sum_, axis=-1), out=self.A)

            # compute batch loss
            if loss or bprop:    
                self.index = np.arange(Y.size)
            
            if loss:
                self.arr = np.log(self.Z_exp_sum_)
                np.add(self.arr, self.Z_max_, out=self.arr)
                np.subtract(self.arr, Z[self.index, ..., Y], out=self.arr)
                self.loss = np.mean(self.arr, axis=0, keepdims=True)

            if accuracy:
                self.AY = np.argmax(self.A, axis=-1)
                np.equal(self.AY, Y, out=self.AY)
                self.accuracy = np.mean(self.AY, axis=0, keepdims=True)

            if bprop:
                # compute derivative of per-example loss
                self.dJdz = np.copy(self.A)
                self.dJdz[self.index, ..., Y] -= 1

                # compute derivatives wrt. parameters
                self.dJdW_ = np.multiply(
                    A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :]
                )
                self.dJdW = np.mean(self.dJdW_, axis=0)
                self.dJdb = np.mean(self.dJdz, axis=0)

            self.run = func

        self.run = func_


class Layer:
    rng = np.random.default_rng()
    def __init__(self, i, j):
        self.width = j
        self.W = self.rng.uniform(low=-.5, high=.5, size=(i, j))
        self.b = self.rng.uniform(low=-.5, high=.5, size=(j,))

    def load_params(self, wpath, bpath):
        self.W = np.load(wpath)
        self.b = np.load(bpath)


class LayerWrapper:
    def __init__(self, layer):
        if layer is not None:
            self.layer = layer
            self.W = self.layer.W
            self.b = self.layer.b


class pipeline:
    def __init__(self, layers_wrapped):
        self.layers = layers_wrapped
        self.layer_in = LayerWrapper(None)
        self.layer_in.act = relu()
        previous = self.layer_in
        for lw in self.layers:
            lw.ff = ff()
            lw.act = relu()
            lw.bprop = bprop()
            lw.previous = previous
            previous.next = lw
            previous = lw

        self.layer_out = self.layers.pop()

        self.model = mlogit()

        self.dJdW_ = None
        self.dJdb_ = None

    def feedforward(self, X):
        self.layer_in.act.A = X
        for lw in self.layers:
            lw.ff.run(lw.previous.act.A, lw.W, lw.b)
            lw.act.run(lw.ff.Z)
        lw = self.layer_out
        lw.ff.run(lw.previous.act.A, lw.W, lw.b)

    def run_model(self, **kwargs):
        lw = self.layer_out
        self.model.run(Z=lw.ff.Z, A_m1=lw.previous.act.A, **kwargs)
        return self.model

    def bprop(self):
        dJdz_p1 = self.model.dJdz
        for lw in self.layers[::-1]:
            lw.bprop.run(
                dJdz_p1=dJdz_p1,
                W_p1=lw.next.W,
                A_m1=lw.previous.act.A,
                s_deriv=lw.act.deriv,
            )
            dJdz_p1 = lw.bprop.dJdz

    @property
    def dJdW(self):
        if self.dJdW_ is None:
            dJdW = [lw.bprop.dJdW for lw in self.layers]
            dJdW.append(self.model.dJdW)
            self.dJdW_ = dJdW
        return self.dJdW_

    @property
    def dJdb(self):
        if self.dJdb_ is None:
            dJdb = [lw.bprop.dJdb for lw in self.layers]
            dJdb.append(self.model.dJdb)
            self.dJdb_ = dJdb
        return self.dJdb_


class Training:
    def __init__(self, layers, training_it, valset, alpha=0.001):
        self.training_pipeline = pipeline([LayerWrapper(layer) for layer in layers])
        self.validation_pipeline = pipeline([LayerWrapper(layer) for layer in layers])
        self.training_it = training_it
        self.valset = valset

        self.W = [layer.W for layer in layers]
        self.b = [layer.b for layer in layers]

        self.optimizer = adam(b1=0.9, b2=0.999, eps=1e-8, alpha=alpha)

        self.completed = False


    def train_minibatch(self, loss=True, accuracy=True, optimize=True):
        batch = next(self.training_it, None)
        if batch is None:
            self.completed = True
            return

        X, Y = batch
        self.training_pipeline.feedforward(X)
        self.training_pipeline.run_model(Y=Y, bprop=True, accuracy=accuracy)
        self.training_pipeline.bprop()

        self.optimizer.run(self.training_pipeline.dJdW, self.training_pipeline.dJdb)

        X, Y = self.valset
        self.validation_pipeline.feedforward(X)
        model = self.validation_pipeline.run_model(Y=Y, bprop=False, accuracy=True)

        if optimize:
            for T, DT in ((self.W, self.optimizer.dW), (self.b, self.optimizer.db)):
                for t, dt in zip(T, DT):
                    np.add(t, dt, out=t)

    def run_batch(self, **model_kwargs):
        batch = next(self.training_it, None)
        if batch is None:
            self.completed = True
            return

        X, Y = batch
        self.training_pipeline.feedforward(X)
        self.training_pipeline.run_model(Y=Y, **model_kwargs)
        self.training_pipeline.bprop()

    def validate(self, **model_kwargs):
        X, Y = self.valset
        self.validation_pipeline.feedforward(X)
        self.validation_pipeline.run_model(Y=Y, **model_kwargs)

    def optimize(self):
        self.optimizer.run(self.training_pipeline.dJdW, self.training_pipeline.dJdb)
        for T, DT in ((self.W, self.optimizer.dW), (self.b, self.optimizer.db)):
                for t, dt in zip(T, DT):
                    np.add(t, dt, out=t)

