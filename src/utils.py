import time
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

colors = ["#e6550d", "#fdd0a2", "#bdbdbd", "#c7e9c0", "#31a354"]
nodes = [0.0, 0.49, 0.5, 0.51, 1.0]
cmap_red_green = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))


class Array:
    def __init__(self, shape, growth_factor=1.5):
        self.shape = shape
        self.len = self.shape[0]
        self.growth_factor = growth_factor
        self.data_ = np.empty(self.shape)
        self.i = 0

    def expand(self):
        newlen = int(self.growth_factor * self.len)
        data_ = np.empty((newlen, *self.shape[1:]))
        data_[: self.len] = self.data_
        self.data_ = data_
        self.len = newlen

    def insert(self, val):
        if self.i == self.len:
            self.expand()
            self.insert(val)
        self.data_[self.i] = val
        self.i += 1

    def clear(self):
        self.i = 0

    @property
    def data(self):
        return self.data_[: self.i]


class Timer:
    def __init__(self):
        self.time = time.time()

    def log(self):
        time_prev = self.time
        self.time = time.time()
        print(self.time - time_prev)


class Iterator:
    def __iter__(self):
        return self


class flatten(Iterator):
    def __init__(self, it, dim):
        if not dim > 0:
            raise ValueError("Dimension must be greater than 0")
        self.dim = dim
        self.its = [iter(it)]

    def __next__(self):
        try:
            while len(self.its) != self.dim:
                self.its.append(iter(next(self.its[-1])))
            return next(self.its[-1])
        except StopIteration:
            if len(self.its) == 1:
                raise
            self.its.pop()
            return next(self)


class epoch_it(Iterator):
    def __init__(self, X, epochs, batchsize):
        self.X = X
        self.epochs = epochs
        self.batchsize = batchsize
        n, _ = self.X.shape
        self.i = 0
        self.m = int(n / self.batchsize)
        self.N = self.epochs * self.m

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


class callback_it(Iterator):
    def __init__(self, it, init_func=None, callback=None):
        self.it = it
        self.init_func = init_func
        self.callback = callback

    def __next__(self):
        if self.init_func is not None:
            self.init_func()
            self.init_func = None
            return None
        if self.callback is not None:
            try:
                return next(self.it)
            except StopIteration:
                self.callback()
                self.callback = None
                return None
        return next(self.it)


class repeat_it(Iterator):
    def __init__(self, func, n):
        self.func = func
        self.n = n
        self.i = 0

    def __next__(self):
        if self.i < self.n:
            self.i += 1
            return self.func()
        else:
            raise StopIteration


def repeat(func, n):
    def fn():
        for _ in range(n):
            func()

    return fn
