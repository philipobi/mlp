import numpy as np
from mlp import MLP, Layer, InputLayer, OutputLayer, MLogit
from utils import Adam
from visualization import MLPVisualization
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from itertools import count
from math import ceil
from utils import epoch_it


def init_training_data(path, batchsize, valsetsize, epochs):

    data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=int)
    valset = data[:valsetsize]
    valset = (valset[:, 1:] / 255, valset[:, 0])
    trainset = data[valsetsize:]

    example_it = epoch_it(trainset, epochs=epochs, batchsize=batchsize)

    N = example_it.N

    it = map(lambda batch: (batch[:, 1:] / 255, batch[:, 0]), example_it)

    return (N, it, valset)


class MLPInterface:
    def __init__(self, mlp):
        self.layers = [LayerInterface(layer) for layer in mlp.layers]


class LayerInterface:
    def __init__(self, layer):
        self._layer = layer
        self.width = self._layer.width

    @property
    def weights(self):
        return self._layer.W.T

    @property
    def activations(self):
        A = self._layer.A
        if len(A.shape) > 1:
            return A[0]
        else:
            return A


class Program:
    batch_windowsize = None

    def __init__(self, model, data_path="data/train.csv"):

        batchsize = 20
        valsetsize = 100
        epochs = 2

        N, self.it, valset = init_training_data(
            data_path,
            batchsize=batchsize,
            valsetsize=valsetsize,
            epochs=epochs,
        )

        self.network = model

        self.network.init_training(batchsize=batchsize, validation_set=valset)

        mlp_interface = MLPInterface(self.network)

        self.visualization = MLPVisualization(mlp_interface, dx=10, dy=3, r=1)

        self.ani = None

        self.visualization.btn_start.on_clicked(self.start)

        plt.show()

    def start(self, _):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.index = []

        if self.batch_windowsize is None:
            self.ymax = 10
            self.visualization.loss_plot.dataLim.y1 = self.ymax
            self.update_plots = self.fixed_plot_update
        else:
            self.update_plots = self.moving_window_plot_update

        self.ani = FuncAnimation(
            self.visualization.fig,
            self.update,
            frames=count(start=1),
            interval=20,
            cache_frame_data=False,
        )

    def fixed_plot_update(self, n_frame):
        xmax = ceil(n_frame / 100) * 100
        self.visualization.train_loss_plot.set_xdata(self.index)
        self.visualization.train_loss_plot.set_ydata(self.train_loss)
        self.visualization.val_loss_plot.set_xdata(self.index)
        self.visualization.val_loss_plot.set_ydata(self.val_loss)

        self.visualization.train_accuracy_plot.set_xdata(self.index)
        self.visualization.train_accuracy_plot.set_ydata(self.train_accuracy)
        self.visualization.val_accuracy_plot.set_xdata(self.index)
        self.visualization.val_accuracy_plot.set_ydata(self.val_accuracy)

        self.visualization.loss_plot.dataLim.x1 = xmax
        self.visualization.accuracy_plot.dataLim.x1 = xmax

    def moving_window_plot_update(self, n_frame):
        self.index = self.index[-self.batch_windowsize :]
        self.train_loss = self.train_loss[-self.batch_windowsize :]
        self.train_accuracy = self.train_accuracy[-self.batch_windowsize :]
        self.val_loss = self.val_loss[-self.batch_windowsize :]
        self.val_accuracy = self.val_accuracy[-self.batch_windowsize :]

        xmin = self.index[0]
        xmax = max(self.batch_windowsize, self.index[-1])

        # update loss plots
        self.visualization.train_loss_plot.set_xdata(self.index)
        self.visualization.train_loss_plot.set_ydata(self.train_loss)

        self.visualization.val_loss_plot.set_xdata(self.index)
        self.visualization.val_loss_plot.set_ydata(self.val_loss)

        self.visualization.loss_plot.dataLim.x0 = xmin
        self.visualization.loss_plot.dataLim.x1 = xmax
        self.visualization.loss_plot.dataLim.y1 = max(
            max(self.train_loss), max(self.val_loss)
        )

        # update accuracy plots
        self.visualization.train_accuracy_plot.set_xdata(self.index)
        self.visualization.train_accuracy_plot.set_ydata(self.train_accuracy)

        self.visualization.val_accuracy_plot.set_xdata(self.index)
        self.visualization.val_accuracy_plot.set_ydata(self.val_accuracy)

        self.visualization.accuracy_plot.dataLim.x0 = xmin
        self.visualization.accuracy_plot.dataLim.x1 = xmax

    def update(self, n_frame):
        try:
            (training_loss, training_accuracy, validation_loss, validation_accuracy) = (
                self.network.train_minibatch(next(self.it))
            )

            self.index.append(n_frame)
            self.train_loss.append(training_loss)
            self.train_accuracy.append(training_accuracy)
            self.val_loss.append(validation_loss)
            self.val_accuracy.append(validation_accuracy)

            self.update_plots(n_frame)

            self.visualization.loss_plot.autoscale_view()
            self.visualization.accuracy_plot.autoscale_view()

            self.visualization.update_weights()

        except StopIteration:
            self.ani.event_source.stop()
            self.ani = None
            self.network.cleanup_training()
            print("Training ended.")


def ModelSpecLarge():
    class DropoutConfig:
        enabled = True
        p_hidden = 0.7
        p_input = 0.8

    return MLP(
        [
            InputLayer(28 * 28),
            Layer(100),
            Layer(100),
            Layer(100),
            Layer(100),
            OutputLayer(10),
        ],
        model=MLogit,
        optimizer=Adam(eps=0.1),
        dropout=DropoutConfig,
    )


def ModelSpecSmall():
    class DropoutConfig:
        enabled = False

    return MLP(
        [
            InputLayer(28 * 28),
            Layer(16),
            Layer(16),
            OutputLayer(10),
        ],
        model=MLogit,
        optimizer=Adam(eps=0.4),
        dropout=DropoutConfig,
    )


if __name__ == "__main__":
    Program(model=ModelSpecSmall())
