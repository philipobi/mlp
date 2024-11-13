import numpy as np


class ConvFilter:
    rng = None

    def __init__(self, shape):
        self.parent = None
        self.child = None
        n, m = shape
        a = np.sqrt(6 / (m + n))
        self.K = self.rng.uniform(low=-a, high=a, size=(n, m))
        self.b = np.array(0.0)

    def append(self, child):
        self.child = child
        self.child.parent = self

    def init_training(self, batchsize):
        _, n, m = self.parent.A.shape
        n1, m1 = self.K.shape
        self.A = np.empty((batchsize, n - n1 + 1, m - m1 + 1))
