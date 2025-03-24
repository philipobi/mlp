import numpy as np


class adam_block:
    b1 = None
    b2 = None
    alpha = None
    eps = None

    def __init__(self):
        def func(g, t):
            m = self.m
            arr0 = self.arr0
            v = self.v
            arr1 = self.arr1

            # m <- b1 * m + (1-b1) * g
            np.multiply(1 - self.b1, g, out=arr0)
            np.multiply(self.b1, m, out=m)
            np.add(m, arr0, out=m)
            # ^m <- m / (1 - b1**t)
            np.divide(m, 1 - self.b1**t, out=arr0)

            # v <- b2 * v + (1-b2) * g^2
            np.square(g, out=arr1)
            np.multiply(1 - self.b2, arr1, out=arr1)
            np.multiply(self.b2, v, out=v)
            np.add(v, arr1, out=v)
            # ^v <- v / (1 - b2**t)
            np.divide(v, 1 - self.b2**t, out=arr1)

            # d <- - a * ^m / (sqrt ^v + eps)
            np.sqrt(arr1, out=arr1)
            np.add(arr1, self.eps, out=arr1)
            np.divide(arr0, arr1, out=arr0)
            np.multiply(-self.alpha, arr0, out=arr0)
            np.copyto(src=arr0, dst=self.d)

        def func_(g, t):
            self.m = np.zeros_like(g)
            self.arr0 = np.empty_like(g)
            self.v = np.zeros_like(g)
            self.arr1 = np.empty_like(g)
            self.d = np.empty_like(g)

            func(g, t)

            self.run = func

        self.run = func_


class adam:
    def __init__(self, b1, b2, alpha, eps):
        adam_block.b1 = b1
        adam_block.b2 = b2
        adam_block.alpha = alpha
        adam_block.eps = eps
        self.t = 0

        def func(dJdW, dJdb):
            self.t += 1
            for tup in [(dJdW, self.W_blocks), (dJdb, self.b_blocks)]:
                for g, block in zip(*tup):
                    block.run(g, self.t)

        def func_(dJdW, dJdb):
            self.W_blocks = [adam_block() for _ in dJdW]
            self.b_blocks = [adam_block() for _ in dJdb]

            func(dJdW, dJdb)

            self.dW = [block.d for block in self.W_blocks]
            self.db = [block.d for block in self.b_blocks]

            self.run = func

        self.run = func_
