import numpy as np
from main import init_training_data
import os.path
from time import sleep


class ff:
    def malloc(self, A, W, b):
        self.Z_ = np.matmul(A, W)
        self.Z = np.add(self.Z_, b)

    def run(self, A, W, b):
        np.matmul(A, W, out=self.Z_)
        np.add(self.Z_, b, out=self.Z)


class bprop:
    def malloc(self, dJdz_p1, W_p1, A_m1, s_deriv):
        self.dJdz_ = np.einsum("...j,...ij->...i", dJdz_p1, W_p1)
        self.dJdz = np.multiply(s_deriv, self.dJdz_)
        self.dJdW_ = np.multiply(A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :])
        self.dJdW = np.sum(self.dJdW_, axis=-3)
        self.dJdb = np.sum(self.dJdz, axis=-2)

    def run(self, dJdz_p1, W_p1, A_m1, s_deriv):
        np.einsum("...j,...ij->...i", dJdz_p1, W_p1, out=self.dJdz_)
        np.multiply(s_deriv, self.dJdz_, out=self.dJdz)
        np.multiply(A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_)
        np.sum(self.dJdW_, axis=-3, out=self.dJdW)
        np.sum(self.dJdz, axis=-2, out=self.dJdb)


class relu:
    def malloc(self, Z):
        self.deriv = np.greater(Z, 0)
        self.A = np.multiply(Z, self.deriv)

    def run(self, Z):
        np.greater(Z, 0, out=self.deriv)
        np.multiply(Z, self.deriv, out=self.A)


class adam_block:
    def malloc(self, g):
        self.d = np.empty_like(g)

        self.m = np.zeros_like(g)
        self.arr0 = np.empty_like(g)

        self.v = np.zeros_like(g)
        self.arr1 = np.empty_like(g)
        return True

    def run(self, g, t, b1, b2, alpha, eps):
        m = self.m
        arr0 = self.arr0
        v = self.v
        arr1 = self.arr1

        # m <- b1 * m + (1-b1) * g
        np.multiply(1 - b1, g, out=arr0)
        np.multiply(b1, m, out=m)
        np.add(m, arr0, out=m)
        # ^m <- m / (1 - b1**t)
        np.divide(m, 1 - b1**t, out=arr0)

        # v <- b2 * v + (1-b2) * g^2
        np.square(g, out=arr1)
        np.multiply(1 - b2, arr1, out=arr1)
        np.multiply(b2, v, out=v)
        np.add(v, arr1, out=v)
        # ^v <- v / (1 - b2**t)
        np.divide(v, 1 - b2**t, out=arr1)

        # d <- - a * ^m / (sqrt ^v + eps)
        np.sqrt(arr1, out=arr1)
        np.add(arr1, eps, out=arr1)
        np.divide(arr0, arr1, out=arr0)
        np.multiply(-alpha, arr0, out=arr0)
        np.copyto(src=arr0, dst=self.d)


class adam:
    def __init__(self, b1, b2, alpha, eps):
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.eps = eps
        self.t = 0

    def malloc(self, dJdW, dJdb):
        self.W_blocks = [
            block for g in dJdW if (block := adam_block()) and block.malloc(g)
        ]
        self.dW = [block.d for block in self.W_blocks]
        self.b_blocks = [
            block for g in dJdb if (block := adam_block()) and block.malloc(g)
        ]
        self.db = [block.d for block in self.b_blocks]

    def run(self, dJdW, dJdb):
        self.t += 1
        for tup in [(dJdW, self.W_blocks), (dJdb, self.b_blocks)]:
            for g, block in zip(*tup):
                block.run(g, self.t, self.b1, self.b2, self.alpha, self.eps)


class mlogit:
    def malloc(self, Z, Y, A_m1):
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

        # compute derivatives wrt. parameters
        self.dJdW_ = np.multiply(A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :])
        self.dJdW = np.sum(self.dJdW_, axis=-3)
        self.dJdb = np.sum(self.dJdz, axis=-2)

    def run(self, Z, Y, A_m1):
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

        # compute derivatives wrt. parameters
        np.multiply(
            A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_
        )
        np.sum(self.dJdW_, axis=-3, out=self.dJdW)
        np.sum(self.dJdz, axis=-2, out=self.dJdb)


class Layer:
    pass


def main():
    # init
    layer_in = Layer()
    layer_in.act = relu()

    dims = [28 * 28, 100, 100, 100, 100, 10]
    layers = []
    path = "params/large_50epochs_90percent"
    previous = layer_in
    for n, (i, j) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        l = Layer()

        l.previous = previous
        previous.next = l

        # l.W = np.load(os.path.join(path, f"W{n}.npy"))
        # l.b = np.load(os.path.join(path, f"b{n}.npy"))
        l.W = np.random.rand(i,j)
        l.b = np.random.rand(j)
        l.ff = ff()
        l.bprop = bprop()
        l.act = relu()
        layers.append(l)

        previous = l

    model = mlogit()

    W = [layer.W for layer in layers]

    b = [layer.b for layer in layers]

    layer_out = layers.pop()

    (N, it, valset) = init_training_data(
        "data/train.csv", batchsize=20, valsetsize=50, epochs=1
    )

    # malloc
    (X, Y) = next(it)
    layer_in.act.A = X
    for layer in layers:
        layer.ff.malloc(layer.previous.act.A, layer.W, layer.b)
        layer.act.malloc(layer.ff.Z)

    layer_out.ff.malloc(layer.act.A, layer_out.W, layer_out.b)

    model.malloc(layer_out.ff.Z, Y, layer.act.A)

    dJdz_p1 = model.dJdz
    for layer in layers[::-1]:
        layer.bprop.malloc(dJdz_p1, layer.next.W, layer.previous.act.A, layer.act.deriv)
        dJdz_p1 = layer.bprop.dJdz

    dJdW = [layer.bprop.dJdW for layer in layers]
    dJdW.append(model.dJdW)

    dJdb = [layer.bprop.dJdb for layer in layers]
    dJdb.append(model.dJdb)

    optimizer = adam(b1=0.9, b2=0.999, eps=1e-8, alpha=0.1)
    optimizer.malloc(dJdW, dJdb)

    # train

    for X, Y in it:
        # inference
        layer_in.act.A = X
        for layer in layers:
            layer.ff.run(layer.previous.act.A, layer.W, layer.b)
            layer.act.run(layer.ff.Z)

        layer_out.ff.run(layer.act.A, layer_out.W, layer_out.b)

        model.run(layer_out.ff.Z, Y, layer.act.A)
        print(model.loss[0], end='\r')

        # bprop
        dJdz_p1 = model.dJdz
        for layer in layers[::-1]:
            layer.bprop.run(
                dJdz_p1, layer.next.W, layer.previous.act.A, layer.act.deriv
            )
            dJdz_p1 = layer.bprop.dJdz

        # optimize params
        optimizer.run(dJdW, dJdb)
        for tup in [(W, optimizer.dW), (b, optimizer.db)]:
            for t, dt in zip(*tup):
                np.add(t, dt, out=t)

        sleep(0.1)


main()
