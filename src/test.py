import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from mlp import MLP, Layer, InputLayer, OutputLayer, MultinoulliML
from utils import Adam


def epoch_it(X, epochs, batchsize):
    n, _ = X.shape
    N = int(n / batchsize)
    for _ in range(epochs):
        for i in range(N):
            yield X[i * batchsize : (i + 1) * batchsize]


network = MLP(
    [InputLayer(28 * 28), Layer(16), Layer(16), OutputLayer(10)],
    model=MultinoulliML(),
    optimizer=Adam(eps=1.0),
)


batchsize = 20
epochs = 1
interval = 40

data = np.loadtxt("data/train.csv", skiprows=1, delimiter=",", dtype=int)
valset = data[:batchsize]
valset = (valset[:, 1:] / 255, valset[:, 0])
trainset = data[batchsize:]
it = epoch_it(trainset, epochs, batchsize=batchsize)

network.init_training(batchsize=batchsize, validation_set=valset)

its = epochs * int(trainset.shape[0] / batchsize)
x = np.arange(its)

y = []


fig, ax = plt.subplots()
batch = next(it)
(X, Y) = (batch[:, 1:] / 255, batch[:, 0])
cost = network.train_minibatch((X, Y))
y.append(cost)
line = ax.plot(x[0], cost)[0]
ax.set(xlim=[0, 10], ylim=[0, cost])


def update(i):
    batch = next(it)
    (X, Y) = (batch[:, 1:] / 255, batch[:, 0])
    cost = network.train_minibatch((X, Y))
    y.append(cost)
    line.set_xdata(x[:i])
    line.set_ydata(y[:i])
    ax.set_xlim(0, i)


ani = FuncAnimation(fig=fig, func=update, frames=its, interval=interval)
plt.show()


"""
it = map(
    lambda line: np.fromstring(line, sep=",", dtype=np.uint8),
    file_it("../data/test.csv", skiplines=1),
)



data = next(it)
n = network.feedforward(data/255)
print(n)
plt.imshow(data.reshape((28,28)), cmap="gray", vmin = 0, vmax = 255)


import numpy as np


class ParentA:
    def func(self, a):
        print("parent a", a)

class ParentB:
    def func(self, a):
        print("parent b", a)

class Derived(ParentA, ParentB):
    def func(self, a):
        super().func(a=a)
        print("derived", a)


Derived().func("hello")





"""
