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


class Adam:
    def __init__(
        self, eps=0.001, r1=0.9, r2=0.999, d=1e-8, clip_threshold=None, clip_stepsize=1
    ):
        self.eps = eps
        self.r1 = r1
        self.r2 = r2
        self.d = d
        self.clip_threshold = clip_threshold
        self.clip_stepsize = clip_stepsize

    def init_training(self, layers):
        self.t = 0
        self.data = []
        for layer in layers:
            sW = np.zeros_like(layer.W)
            sW_ = np.empty_like(layer.W)
            rW = np.zeros_like(layer.W)
            rW_ = np.empty_like(layer.W)

            sb = np.zeros_like(layer.b)
            sb_ = np.empty_like(layer.b)
            rb = np.zeros_like(layer.b)
            rb_ = np.empty_like(layer.b)
            self.data.append((layer, (sW, sW_, rW, rW_), (sb, sb_, rb, rb_)))

    @property
    def _gradient_norm(self):
        nrm2 = 0
        for layer, _, _ in self.data:
            nrm2 += np.square(np.linalg.norm(layer.dJdW))
            nrm2 += np.square(np.linalg.norm(layer.dJdb))
        return np.sqrt(nrm2)

    def _scale_gradients(self, r):
        for layer, _, _ in self.data:
            np.multiply(layer.dJdW, r, out=layer.dJdW)
            np.multiply(layer.dJdb, r, out=layer.dJdb)

    def _clip_gradients(self):
        nrm = self._gradient_norm
        if np.isinf(nrm) or np.isnan(nrm):
            for layer, _, _ in self.data:
                layer.dJdW = np.random.rand(*layer.dJdW.shape) - 0.5
                layer.dJdb = np.random.rand(*layer.dJdb.shape) - 0.5
            self._scale_gradients(self.clip_stepsize / self._gradient_norm)
        elif nrm > self.clip_threshold:
            self._scale_gradients(self.clip_stepsize / nrm)

    def update_params(self):
        self.t += 1
        r1, r2, t = (self.r1, self.r2, self.t)

        if self.clip_threshold is not None:
            self._clip_gradients()

        for layer, W_, b_ in self.data:
            for (s, s_, r, r_), g, p in (
                (W_, layer.dJdW, layer.W),
                (b_, layer.dJdb, layer.b),
            ):
                np.multiply(r1, s, out=s)
                np.multiply(1 - r1, g, out=s_)
                np.add(s, s_, out=s)

                np.multiply(r2, r, out=r)
                np.square(g, out=r_)
                np.multiply(1 - r2, r_, out=r_)
                np.add(r, r_, out=r)

                np.divide(s, 1 - np.power(r1, t), out=s)
                np.divide(r, 1 - np.power(r2, t), out=r)

                np.sqrt(r, out=r)
                np.add(r, self.d, out=r)
                np.multiply(s, -self.eps, out=s)
                np.divide(s, r, out=s)
                np.add(p, s, out=p)


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
