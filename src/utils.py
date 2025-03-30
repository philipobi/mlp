import numpy as np


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