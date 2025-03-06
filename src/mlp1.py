import numpy as np
from main import init_training_data

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
        
        #compute batch loss
        self.Z_max_ = np.max(Z, axis=-1)
        self.A = np.subtract(Z, self.Z_max_)
        np.exp(self.A, out=self.A)
        self.Z_exp_sum_ = np.sum(self.A, axis=-1)
        self.arr = np.log(self.Z_exp_sum_)
        np.add(self.arr, self.Z_max_, out=self.arr)
        np.subtract(self.arr, Z[..., self.index_, Y], out=self.arr)
        self.loss = np.sum(self.arr, axis=-1)
        np.divide(self.loss, self.N, out=self.loss)

        #compute activations
        np.divide(self.A, self.Z_exp_sum_, out=self.A)

        #compute derivative of per-example loss
        self.dJdz = np.copy(self.A)
        self.dJdz[..., self.index_, Y] -= 1

    def run(self, Z, Y):
        #compute batch loss
        np.max(Z, axis=-1, out=self.Z_max_)
        np.subtract(Z, self.Z_max_, out=self.A)
        np.exp(self.A, out=self.A)
        np.sum(self.A, axis=-1, out=self.Z_exp_sum_)
        np.log(self.Z_exp_sum_j, out=self.arr)
        np.add(self.arr, self.Z_max_, out=self.arr)
        np.subtract(self.arr, Z[..., self.index_, Y], out=self.arr)
        np.sum(self.arr, axis=-1, out=self.loss)
        np.divide(self.loss, self.N, out=self.loss)

        #compute activations
        np.divide(self.A, self.Z_exp_sum_, out=self.A)

        #compute derivative of per-example loss
        np.copyto(src=self.A, dst=self.dJdz)
        self.dJdz[..., self.index_, Y] -= 1

class Layer:
    pass

def main():
    dims = [28*28, 100, 100, 100, 10]
    layers = []
    for i,j in zip(dims[:-1], dims[1:]):
        l = Layer()
        l.W = np.random.rand(i,j)
        l.b = np.random.rand(j)
        l.ff = ff()
        l.bprop = bprop()
        l.act = relu()
        layers.append(l)
    
    
    (N, it, valset) = init_training_data("data/train.csv", batchsize=50, valsetsize=50, epochs=20)
    (X, Y) = next(it)
    A = X
    for layer in layers[:-1]:
        layer.ff.malloc(A, layer.W, layer.b)
        layer.act.malloc(layer.ff.Z)
        A = layer.act.A

main()