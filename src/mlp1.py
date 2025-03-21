import numpy as np
from main import init_training_data
import os.path
from time import sleep


class ff:
    def __init__(self):
        def func(A, W, b):
            np.matmul(A, W, out=self.Z_)
            np.add(self.Z_, b, out=self.Z)

        def func_(A, W, b):
            self.Z_ = np.matmul(A, W)
            self.Z = np.add(self.Z_, b)
            self.run = func

        self.run = func_


class bprop:
    def __init__(self):

        def func(dJdz_p1, W_p1, A_m1, s_deriv):
            np.einsum("...j,...ij->...i", dJdz_p1, W_p1, out=self.dJdz_)
            np.multiply(s_deriv, self.dJdz_, out=self.dJdz)
            np.multiply(
                A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_
            )
            np.sum(self.dJdW_, axis=-3, out=self.dJdW)
            np.sum(self.dJdz, axis=-2, out=self.dJdb)

        def func_(dJdz_p1, W_p1, A_m1, s_deriv):
            self.dJdz_ = np.einsum("...j,...ij->...i", dJdz_p1, W_p1)
            self.dJdz = np.multiply(s_deriv, self.dJdz_)
            self.dJdW_ = np.multiply(
                A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :]
            )
            self.dJdW = np.sum(self.dJdW_, axis=-3)
            self.dJdb = np.sum(self.dJdz, axis=-2)

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


class adam_block:
    def __init__(self):
        def func(g, t, b1, b2, alpha, eps):
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

        def func_(g, t, b1, b2, alpha, eps):
            self.m = np.zeros_like(g)
            self.arr0 = np.empty_like(g)
            self.v = np.zeros_like(g)
            self.arr1 = np.empty_like(g)
            self.d = np.empty_like(g)

            func(g, t, b1, b2, alpha, eps)

            self.run = func

        self.run = func_


class adam:
    def __init__(self, b1, b2, alpha, eps):
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.eps = eps
        self.t = 0

        def func(dJdW, dJdb):
            self.t += 1
            for tup in [(dJdW, self.W_blocks), (dJdb, self.b_blocks)]:
                for g, block in zip(*tup):
                    block.run(g, self.t, self.b1, self.b2, self.alpha, self.eps)

        def func_(dJdW, dJdb):
            self.W_blocks = [adam_block() for _ in dJdW]
            self.b_blocks = [adam_block() for _ in dJdb]

            func(dJdW, dJdb)

            self.dW = [block.d for block in self.W_blocks]
            self.db = [block.d for block in self.b_blocks]

            self.run = func

        self.run = func_


class mlogit:
    def __init__(self):

        def func(Z, Y, A_m1, bprop=True):
            # compute batch loss
            np.max(Z, axis=-1, out=self.Z_max_)
            np.subtract(Z, np.expand_dims(self.Z_max_, axis=-1), out=self.A)
            np.exp(self.A, out=self.A)
            np.sum(self.A, axis=-1, out=self.Z_exp_sum_)
            np.log(self.Z_exp_sum_, out=self.arr)
            np.add(self.arr, self.Z_max_, out=self.arr)
            np.subtract(self.arr, Z[..., self.index_, Y], out=self.arr)
            np.sum(self.arr, axis=-1, keepdims=True, out=self.loss)
            np.divide(self.loss, self.N, out=self.loss)

            # compute activations
            np.divide(self.A, np.expand_dims(self.Z_exp_sum_, axis=-1), out=self.A)

            if bprop:
                # compute derivative of per-example loss
                np.copyto(src=self.A, dst=self.dJdz)
                self.dJdz[..., self.index_, Y] -= 1

                # compute derivatives wrt. parameters
                np.multiply(
                    A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :], out=self.dJdW_
                )
                np.sum(self.dJdW_, axis=-3, out=self.dJdW)
                np.sum(self.dJdz, axis=-2, out=self.dJdb)

        def func_(Z, Y, A_m1, brop=True):
            ###
            self.N = Y.size
            self.index_ = np.arange(self.N)
            ###

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

            if bprop:
                # compute derivative of per-example loss
                self.dJdz = np.copy(self.A)
                self.dJdz[..., self.index_, Y] -= 1

                # compute derivatives wrt. parameters
                self.dJdW_ = np.multiply(
                    A_m1[..., np.newaxis], self.dJdz[..., np.newaxis, :]
                )
                self.dJdW = np.sum(self.dJdW_, axis=-3)
                self.dJdb = np.sum(self.dJdz, axis=-2)

            self.run = func

        self.run = func_


class Layer:
    def __init__(self, i, j):
        self.W = np.random.rand(i, j)
        self.b = np.random.rand(j)


class ProjectionAxis:
    def __init__(self, arr, pos, num, d):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.d = d
        self.redraw_ax(all=True)

    @property
    def in_range(self):
        x0 = self.arr[..., *self.pos]
        return x0 >= self.xmin and x0 <= self.xmax

    def redraw_ax(self, all=False):
        if all:
            self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        x0 = self.arr[..., *self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[:, ..., *self.pos] = np.linspace(self.xmin, self.xmax, num=self.num)

class LayerWrapper:
    def __init__(self, layer):
        self.layer = layer

class ProjectionLayer(LayerWrapper):
    def __init__(self, layer, W=None, b=None):
        super().__init__(layer)
        
        self.redraw_hooks = []
        self.ax_W = W
        self.ax_b = b

        if W is None:
            self.W_ = np.copy(self.layer.W)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.W, dst=self.W_))
            self.W_getter = lambda: self.W_
        else:
            self.redraw_hooks.append(lambda: W.redraw_ax(all=True))
            self.W_getter = lambda: W.arr_ax

        if b is None:
            self.b_ = np.copy(self.layer.b)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.b, dst=self.b_))
            self.b_getter = lambda: self.b_
        else:
            self.redraw_hooks.append(lambda: b.redraw_ax(all=True))
            self.b_getter = lambda: b.arr_ax

    def redraw_all(self):
        for fn in self.redraw_hooks:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @property
    def b(self):
        return self.b_getter()


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y
        self.model = mlogit()

        self.axes = []
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.layers = layers[:-1]
        self.layer_out = layers[-1]

        for layer in self.layers:
            layer.ff = ff()
            layer.act = relu()

        self.layer_out.ff = ff()

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        A = self.X
        for layer in self.layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(ff.Z)
            A = layer.act.A

        layer = self.layer_out
        layer.ff.run(A, layer.W, layer.b)
        self.model.run(layer.Z, self.Y, A, bprop=False)

    @property
    def grid(self):
        return self.model.loss

class Training:
    def __init__(self, layers):
        self.layers = [LayerWrapper(layer) for layer in layers]
        

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
        l.W = np.random.rand(i, j)
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

    optimizer = adam(b1=0.9, b2=0.999, eps=1e-8, alpha=0.1)

    dJdW = []
    dJdb = []

    # train
    for X, Y in it:
        # inference
        layer_in.act.A = X
        for layer in layers:
            layer.ff.run(layer.previous.act.A, layer.W, layer.b)
            layer.act.run(layer.ff.Z)

        layer_out.ff.run(layer.act.A, layer_out.W, layer_out.b)

        model.run(layer_out.ff.Z, Y, layer.act.A)
        print(model.loss[0], end="\r")

        # bprop
        dJdz_p1 = model.dJdz
        for layer in layers[::-1]:
            layer.bprop.run(
                dJdz_p1, layer.next.W, layer.previous.act.A, layer.act.deriv
            )
            dJdz_p1 = layer.bprop.dJdz

        # optimize params
        if not dJdW and not dJdb:
            dJdW.extend([layer.bprop.dJdW for layer in layers])
            dJdW.append(model.dJdW)
            dJdb.extend([layer.bprop.dJdb for layer in layers])
            dJdb.append(model.dJdb)

        optimizer.run(dJdW, dJdb)
        for tup in [(W, optimizer.dW), (b, optimizer.db)]:
            for t, dt in zip(*tup):
                np.add(t, dt, out=t)

        sleep(0.1)


main()
