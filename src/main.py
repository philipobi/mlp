import numpy as np
from mlp import Layer, Training
from visualization import MLPVisualization, SimplePlot
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from utils import epoch_it, callback_it, Timer
from projection import ProjectionView, ProjectionLayer, ProjectionAxis
from time import sleep


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
    i = 0
    n_examples = None

    @classmethod
    def next_example(cls):
        cls.i += 1
        if cls.i == cls.n_examples:
            cls.i = 0

    def __init__(self, layer_wrapped, width=None):
        self.lw = layer_wrapped
        self.width = width or self.lw.layer.width
        self.A_getter = lambda: self.lw.act.A

    @property
    def weights(self):
        return self.lw.layer.W.T

    @property
    def activation(self):
        return self.A_getter()[self.i]


class Program:
    batch_windowsize = None

    def __init__(self, data_path="data/train.csv"):

        path = "params/small_50epochs_93percent"
        dims = [28 * 28, 16, 16, 10]
        layers = [Layer(i, j) for i, j in zip(dims[:-1], dims[1:])]
        if 0:
            for i, layer in enumerate(layers, start=1):
                layer.load_params(
                    wpath=path + f"/W{i}.npy",
                    bpath=path + f"/b{i}.npy",
                )

        self.empty_img = np.zeros((28, 28))

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

        LayerInterface.n_examples = valsetsize

        # hook up layer interfaces for visualization to layer wrappers of validation pipeline
        validation_pipeline = self.training.validation_pipeline
        layer_in = LayerInterface(validation_pipeline.layer_in, width=dims[0])
        layer_in.A_getter = lambda: layer_in.lw.act.A
        self.layer_in = layer_in
        layer_out = LayerInterface(validation_pipeline.layer_out)
        layer_out.A_getter = lambda: validation_pipeline.model.A
        interface_layers = [
            layer_in,
            *[
                LayerInterface(layer)
                for layer in self.training.validation_pipeline.layers
            ],
            layer_out,
        ]

        MLPVisualization.maxwidth = 10
        self.visualization = MLPVisualization(interface_layers, dx=10, dy=5, r=2)
        self.visualization.layers[-1].normalize_activations = False
        for i, node in enumerate(self.visualization.layers[-1].nodes):
            node.add_label(f"{i}")

        # Set up descent projection
        X, Y = valset
        proj_layers = [ProjectionLayer(layer) for layer in layers]
        proj_layer = proj_layers[-1]
        proj_layer.add_axes(
            b=(ProjectionAxis(arr=proj_layer.layer.b, pos=(0,), num=100),),
            W=(ProjectionAxis(arr=proj_layer.layer.W, pos=(7, 0), num=100),),
        )
        self.projection_view = ProjectionView(
            self.visualization.ax_loss_projection,
            proj_layers,
            X=X,
            Y=Y,
            update_interval=100,
        )

        # Set up accuracy plot
        self.accuracy_plot = SimplePlot(
            ax=self.visualization.accuracy_plot,
            xlims=(0, 100),
            ylims=(0, 1),
            n_plots=2,
            plot_kwargs=(
                dict(label="Training Accuracy", color="blue"),
                dict(label="Validation Accuracy", color="red"),
            ),
        )

        self.ani = None

        self.ani_running = False

        self.visualization.btn_start.on_clicked(self.btn_start_callback)

        self.plot_img(self.empty_img)

        # make model allocate memory for accuracy calculation
        self.training.run_batch(bprop=True, accuracy=True)

        plt.show()

    def plot_img(self, X):
        self.visualization.ax_img.imshow(
            cmap="gray_r",
            vmin=0,
            vmax=1,
            X=X,
        )

    def animate_ff(self):
        return callback_it(
            init_func=lambda: (
                self.visualization.ax_loss_projection.set_visible(False),
                self.visualization.ax_accuracy.set_visible(False),
                self.plot_img(np.reshape(self.layer_in.activation, shape=(28, 28))),
                self.visualization.switch_cmap(),
            ),
            it=self.visualization.animate_ff(),
            callback=lambda: (
                sleep(3),
                LayerInterface.next_example(),
                self.visualization.clear_activations(),
                self.plot_img(self.empty_img),
                self.visualization.switch_cmap(),
                self.visualization.ax_loss_projection.set_visible(True),
                self.visualization.ax_accuracy.set_visible(True),
            ),
        )

    def animate_main(self):
        while 1:
            for _ in range(200):
                N = 5
                for i in range(N):
                    if self.training.completed:
                        return
                    self.training.run_batch(bprop=True, accuracy=(i == N - 1))
                    self.training.optimize()
                self.i_training += N
                self.training.validate(accuracy=True)

                model = self.training.training_pipeline.model
                training_accuracy = model.accuracy[0]
                model = self.training.validation_pipeline.model
                validation_accuracy = model.accuracy[0]

                self.accuracy_plot.update(
                    self.i_training, training_accuracy, validation_accuracy
                )
                self.visualization.update_weights()
                self.projection_view.draw_frame()

                yield

            yield from self.animate_ff()

    def btn_start_callback(self, _):
        if self.ani is None:
            self.ani_running = True
            self.i_training = 1
            self.ani = FuncAnimation(
                fig=self.visualization.fig,
                func=lambda _: None,
                frames=self.animate_main(),
                cache_frame_data=False,
                interval=10,
                repeat=False,
            )
            return

        if self.ani_running:
            self.ani_running = False
            self.ani.event_source.stop()
        else:
            self.ani_running = True
            self.ani.event_source.start()


if __name__ == "__main__":
    Program()
