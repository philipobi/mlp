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
