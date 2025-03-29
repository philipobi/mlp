import numpy as np
from mlp import Layer, Training
from visualization import MLPVisualization
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from math import ceil
from utils import epoch_it
from itertools import count, cycle

def init_training_data(path, batchsize, valsetsize, epochs):

    data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=int)
    valset = data[:valsetsize]
    valset = (valset[:, 1:] / 255, valset[:, 0])
    trainset = data[valsetsize:]

    example_it = epoch_it(trainset, epochs=epochs, batchsize=batchsize)

    N = example_it.N

    it = map(lambda batch: (batch[:, 1:] / 255, batch[:, 0]), example_it)

    return (N, it, valset)

class LayerInterface:
    def __init__(self, layer_wrapped, width=None):
        self.lw = layer_wrapped
        self.width = width or self.lw.layer.width
        self.A_getter = lambda: self.lw.act.A

    @property
    def weights(self):
        return self.lw.layer.W.T

    @property
    def activation(self):
        return self.A_getter()[0]

class Program:
    batch_windowsize = None

    def __init__(self, data_path="data/train.csv"):

        dims = [28 * 28, 16, 16, 10]
        layers = [Layer(i, j) for i, j in zip(dims[:-1], dims[1:])]
        
        batchsize = 20
        valsetsize = 100
        epochs = 2

        N, self.it, valset = init_training_data(
            data_path,
            batchsize=batchsize,
            valsetsize=valsetsize,
            epochs=epochs,
        )

        self.training = Training(layers, self.it, valset)
        validation_pipeline = self.training.validation_pipeline
        layer_in = LayerInterface(validation_pipeline.layer_in, width=dims[0])
        layer_in.A_getter = lambda: layer_in.lw.act.A
        layer_out = LayerInterface(validation_pipeline.layer_out)
        layer_out.A_getter = lambda: validation_pipeline.model.A
        layers = [
            layer_in,
            *[LayerInterface(layer) for layer in self.training.validation_pipeline.layers],
            layer_out
            ]

        MLPVisualization.maxwidth = 11
        self.visualization = MLPVisualization(layers, dx=10, dy=5, r=2)
        self.visualization.layers[-1].normalize_activations = False
        for i, node in enumerate(self.visualization.layers[-1].nodes):
            node.add_label(f"{i}")

        self.ani = None

        self.visualization.btn_start.on_clicked(self.start)

        self.examples = cycle([img for img in valset[0]])

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
        if not self.training.completed:
            self.training.train_minibatch()

            model = self.training.training_pipeline.model
            training_loss = model.loss[0]
            training_accuracy = model.accuracy[0]
            model = self.training.validation_pipeline.model
            validation_loss = model.loss[0]
            validation_accuracy = model.accuracy[0]

            self.index.append(n_frame)
            self.train_loss.append(training_loss)
            self.train_accuracy.append(training_accuracy)
            self.val_loss.append(validation_loss)
            self.val_accuracy.append(validation_accuracy)

            self.update_plots(n_frame)

            self.visualization.loss_plot.autoscale_view()
            self.visualization.accuracy_plot.autoscale_view()

            self.visualization.update_weights()
        else:
            self.ani.event_source.stop()
            self.ani = None
            print("Training ended.")


if __name__ == "__main__":
    Program()
