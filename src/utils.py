import numpy as np

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
