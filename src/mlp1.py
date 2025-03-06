import numpy as np
from main import init_training_data
import os.path

class ff:
    def malloc(self, A, W, b):
        self.Z_ = np.matmul(A, W)
        self.Z = np.add(self.Z_, b)

    def run(self, A, W, b):
        np.matmul(A, W, out=self.Z_)
        np.add(self.Z_, b, out=self.Z)


class bprop:
    def malloc(self, J1, W1, s_deriv):
        self.dJdz_ = np.einsum("...j,...ij->...i", J1, W1)
        self.dJdz = np.multiply(s_deriv, self.dJdz_)

    def run(self, J1, W1, s_deriv):
        np.einsum("...j,...ij->...i", J1, W1, out=self.dJdz_)
        np.multiply(s_deriv, self.dJdz_, out=self.dJdz)


class relu:
    def malloc(self, Z):
        self.deriv = np.greater(Z, 0)
        self.A = np.multiply(Z, self.deriv)

    def run(self, Z):
        np.greater(Z, 0, out=self.deriv)
        np.multiply(Z, self.deriv, out=self.A)


class mlogit:
    def malloc(self, Z, Y):
        self.N = Y.size
        self.index_ = np.arange(self.N)

        # compute batch loss
        self.Z_max_ = np.max(Z, axis=-1)
        self.A = np.subtract(Z, np.expand_dims(self.Z_max_, axis=-1))
        np.exp(self.A, out=self.A)
        self.Z_exp_sum_ = np.sum(self.A, axis=-1)
        self.arr = np.log(self.Z_exp_sum_)
        np.add(self.arr, self.Z_max_, out=self.arr)
        np.subtract(self.arr, Z[..., self.index_, Y], out=self.arr)
        self.loss = np.sum(self.arr, axis=-1, keepdims=True)    
        np.divide(self.loss, self.N, out=self.loss)

        # compute activations
        np.divide(self.A, np.expand_dims(self.Z_exp_sum_, axis=-1), out=self.A)

        # compute derivative of per-example loss
        self.dJdz = np.copy(self.A)
        self.dJdz[..., self.index_, Y] -= 1

    def run(self, Z, Y):
        # compute batch loss
        np.max(Z, axis=-1, out=self.Z_max_)
        np.subtract(Z, np.expand_dims(self.Z_max_, axis=-1), out=self.A)
        np.exp(self.A, out=self.A)
        np.sum(self.A, axis=-1, out=self.Z_exp_sum_)
        np.log(self.Z_exp_sum_, out=self.arr)
        np.add(self.arr, self.Z_max_, out=self.arr)
        np.subtract(self.arr, Z[..., self.index_, Y], out=self.arr)
        np.sum(self.arr, axis=-1, out=self.loss, keepdims=True)
        np.divide(self.loss, self.N, out=self.loss)

        # compute activations
        np.divide(self.A, np.expand_dims(self.Z_exp_sum_, axis=-1), out=self.A)

        # compute derivative of per-example loss
        np.copyto(src=self.A, dst=self.dJdz)
        self.dJdz[..., self.index_, Y] -= 1


class Layer:
    pass


def main():
    # init
    dims = [28 * 28, 100, 100, 100, 100, 10]
    layers = []
    path = "params/large_50epochs_90percent"
    for n, (i, j) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        l = Layer()
        l.W = np.load(os.path.join(path, f"W{n}.npy"))
        l.b = np.load(os.path.join(path, f"b{n}.npy"))
        l.ff = ff()
        l.bprop = bprop()
        l.act = relu()
        layers.append(l)

    (N, it, valset) = init_training_data(
        "data/train.csv", batchsize=1, valsetsize=50, epochs=20
    )

    layer_out = layers.pop()

    # malloc
    (X, Y) = next(it)
    A = X
    for layer in layers:
        layer.ff.malloc(A, layer.W, layer.b)
        layer.act.malloc(layer.ff.Z)
        A = layer.act.A

    layer_out.ff.malloc(A, layer_out.W, layer_out.b)
    layer_out.model = mlogit()
    layer_out.model.malloc(layer_out.ff.Z, Y)

    W1 = layer_out.W
    J1 = layer_out.model.dJdz
    for layer in layers[::-1]:
        layer.bprop.malloc(J1, W1, layer.act.deriv)
        W1 = layer.W
        J1 = layer.bprop.dJdz

    # run
    # inference
    for _ in range(20):
        (X, Y) = next(it)
        A = X
        for layer in layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(layer.ff.Z)
            A = layer.act.A

        layer_out.ff.run(A, layer_out.W, layer_out.b)
        layer_out.model.run(layer_out.ff.Z, Y)

        print(np.argmax(layer_out.model.A, axis=-1), Y)


    return
    # bprop
    W1 = layer_out.W
    J1 = layer_out.model.dJdz
    for layer in layers[::-1]:
        layer.bprop.run(J1, W1, layer.act.deriv)
        W1 = layer.W
        J1 = layer.bprop.dJdz

main()
