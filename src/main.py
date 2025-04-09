import numpy as np
from mlp import Layer, Training
from visualization import MLPVisualization, Node, WindowedPlot
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from utils import epoch_it, task_it, Timer
from projection import ProjectionView, ProjectionLayer, ProjectionAxis
from time import sleep
from itertools import chain
from matplotlib import ticker

rng = np.random.default_rng(12345)

def init_training_data(path, batchsize, valsetsize, epochs):

    data = np.loadtxt(path, skiprows=1, delimiter=",", dtype=int)
    rng.shuffle(data)

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
            cls.reset_counter()
    
    @classmethod
    def reset_counter(cls):
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

    def __init__(self, data_path="data/train.csv"):

        Layer.rng = rng

        MLPVisualization.maxwidth = 10

        path = "params/small_50epochs_93percent"
        dims = [28 * 28, 10, 10, 10, 10]
        # dims = [28 * 28, 16, 16, 10]
        self.layers = [Layer(i, j) for i, j in zip(dims[:-1], dims[1:])]
        if 0:
            for i, layer in enumerate(self.layers, start=1):
                layer.load_params(
                    wpath=path + f"/W{i}.npy",
                    bpath=path + f"/b{i}.npy",
                )

        batchsize = 20
        valsetsize = 100
        epochs = 10

        N, self.it, self.valset = init_training_data(
            data_path,
            batchsize=batchsize,
            valsetsize=valsetsize,
            epochs=epochs,
        )

        self.training = Training(self.layers, self.it, self.valset, alpha=0.002)

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

        self.visualization = MLPVisualization(interface_layers, dx=10, dy=5, r=2)

        # set up node graph
        self.visualization.layers[-1].normalize_activations = False
        graph = self.visualization.graph
        r = Node.radius
        for i, node in enumerate(self.visualization.layers[-1].nodes):
            (x, y) = node.xy
            graph.annotate(str(i), (x + 2.0 * r, y), va="center")

        # Set up descent projection
        self.projection_view = None
        self.reset_projection_view()
        ax = self.projection_view.ax
        ax.set_xlabel(r"$W^{(4)}_{8 1}$")
        ax.set_ylabel(r"$b^{(4)}_1$")
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)

        # Set up accuracy plot
        self.accuracy_plot = WindowedPlot(
            ax=self.visualization.ax_accuracy,
            ylims=(0, 1),
            n_plots=2,
            windowsize=300,
            plot_kwargs=(
                dict(label="Training acc.", color="blue"),
                dict(label="Validation acc.", color="red"),
            ),
        )
        ax = self.accuracy_plot.ax
        ax.set_xlabel("Batches")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0%}"))
        ax.set_title("Accuracy", loc="left")

        self.ani = None

        self.ani_running = False

        self.visualization.btn_start.on_clicked(self.btn_start_callback)
        self.visualization.btn_reset.on_clicked(self.reset_projection_view)

        self.img = self.visualization.ax_img.imshow(
            cmap="gray_r",
            vmin=0,
            vmax=1,
            X=np.zeros((28, 28)),
        )

        # make model allocate memory for accuracy calculation
        self.training.run_batch(bprop=True, accuracy=True)

        self.visualization.fig.legend(loc="upper left")

        plt.show()

    def reset_projection_view(self, _=None):
        ax = self.visualization.ax_loss_projection
        if self.projection_view is not None:
            ax.clear()
        self.proj_layers = [ProjectionLayer(layer) for layer in self.layers]
        proj_layer = self.proj_layers[-1]
        proj_layer.add_axes(
            b=(ProjectionAxis(arr=proj_layer.layer.b, pos=(0,), num=100),),
            W=(ProjectionAxis(arr=proj_layer.layer.W, pos=(7, 0), num=100),),
        )
        X, Y = self.valset
        self.projection_view = ProjectionView(
            ax,
            self.proj_layers,
            X=X,
            Y=Y,
            update_interval=20,
        )

    def animate_ff_wrapper(self, n=1):
        # fmt: off
        yield from task_it(
            lambda: (
                self.visualization.ax_loss_projection.set_visible(False),
                self.visualization.ax_accuracy.set_visible(False),
                self.visualization.set_div_alpha(0),
                self.visualization.text_validating.set_alpha(1),
            ),
            lambda: (
                sleep(1),
                self.visualization.switch_cmap(),
            ),
        )
        for _ in range(n):
            yield from self.animate_ff()
        yield from task_it(
            lambda: (
                sleep(1),
                self.visualization.switch_cmap(),
            ),
            lambda: (
                sleep(1),
                self.visualization.ax_loss_projection.set_visible(True),
                self.visualization.ax_accuracy.set_visible(True),
                self.visualization.text_validating.set_alpha(0),
                self.visualization.set_div_alpha(1),
            ),
            lambda: sleep(1)
        )
        # fmt: on

    def animate_ff(self):
        # fmt:off
        return chain(
            task_it(
                lambda: (
                    sleep(1),
                    self.img.set_data(np.reshape(self.layer_in.activation, shape=(28, 28))),
                    self.visualization.ax_img.set_visible(True)
                ),
                lambda: sleep(1),
            ),
            
            self.visualization.animate_ff(),

            task_it(
                lambda: (
                    sleep(1),
                    LayerInterface.next_example(),
                    self.visualization.clear_activations(),
                    self.visualization.ax_img.set_visible(False)
                ),
                lambda: sleep(1)
            ),
        )
        # fmt:on

    def animate_main(self):
        steps = [(50, 10), (200, 1), (200, 2), (150, 5)]
        init = True
        while 1:
            if steps:
                (M, n) = steps.pop(0)
            for _ in range(M):
                N = 5
                for i in range(N):
                    if self.training.completed:
                        print("Training completed")
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

                sleep(0.06)

            yield from self.animate_ff_wrapper(n = n)

            if init:
                init = False
                LayerInterface.reset_counter()

    def btn_start_callback(self, _):
        if self.ani is None:
            self.ani_running = True
            self.i_training = 1
            self.ani = FuncAnimation(
                fig=self.visualization.fig,
                func=lambda _: None,
                frames=self.animate_main(),
                cache_frame_data=False,
                interval=6,
                repeat=False,
                blit=False,
            )
            return

        if self.ani_running:
            layer = self.layers[-1]
            print(layer.W[7, 0], layer.b[0])
            self.ani_running = False
            self.ani.event_source.stop()
        else:
            self.ani_running = True
            self.ani.event_source.start()


if __name__ == "__main__":
    Program()
