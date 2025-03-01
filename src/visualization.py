import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    SymLogNorm,
    ListedColormap,
)
from matplotlib.widgets import TextBox, Button
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import numpy as np
import threading

# fmt:off
# red -> green
colors= [
    (0.62745098, 0.00784314, 0.12156863,),
    (0.6674356,  0.00984237, 0.10557478,),
    (0.70695886, 0.01214917, 0.08981161,),
    (0.73710111, 0.02291426, 0.07904652,),
    (0.76509035, 0.03260285, 0.06874279,),
    (0.7952326,  0.04121492, 0.05582468,),
    (0.82537486, 0.07128028, 0.07128028,),
    (0.86412918, 0.16816609, 0.16816609,),
    (0.90025515, 0.26003441, 0.26003441,),
    (0.93940093, 0.3622484,  0.3622484,),
    (0.36123713, 0.80906023, 0.28369353,),
    (0.26554744, 0.75904062, 0.20757673,),
    (0.17723952, 0.71164937, 0.13856209,),
    (0.08250673, 0.65997693, 0.06535948,),
    (0.0544406,  0.61637832, 0.05582468,),
    (0.04798155, 0.57116494, 0.06874279,),
    (0.04875048, 0.53425606, 0.08750481,),
    (0.0509035,  0.49550173, 0.10903499,),
    (0.0413687,  0.45182622, 0.11749327,),
    (0.03137255, 0.40784314, 0.1254902,),
]
# fmt:on
cmap = ListedColormap(colors=colors)
cmap = mpl.colormaps["viridis"]


def parse_float(s, default=None):
    try:
        return float(s)
    except ValueError:
        return default


def fmt_accuracy(train_accuracy, val_accuracy):
    return f"Accuracy: Train {train_accuracy:0.2f}, Validation {val_accuracy:0.2f}"


class Node:
    radius = None

    def __init__(self, i, y_ord, xy=(0, 0), **kwargs):
        self.i = i
        self.y_ord = y_ord
        self.xy_ = xy

        if kwargs.get("radius", None) is None:
            kwargs["radius"] = self.radius

        self.node = Circle(xy=self.xy_, **kwargs)
        kwargs["radius"] = 0.0
        self.activation = Circle(xy=self.xy_, **kwargs)

    @property
    def xy(self):
        return self.xy_

    @xy.setter
    def xy(self, value):
        self.xy_ = value
        self.node.center = self.xy_
        self.activation.center = self.xy_


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.weight = 0

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class LayerVisualization:
    def default_callback(self, event):
        pass

    def __init__(self, layer):
        self.layer = layer
        self.width = layer.width
        self.n = None

        self.overfull = False
        self.overfull_text = None

        self.nodes = []
        self.incoming_edges = []
        self.activation_circles = []
        self._patch_ca = None

        self._visible_edge_weights = lambda W: None
        self._visible_activations = lambda A: None
        self.normalize_activations = True

    def add_nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes.extend(nodes)
        self.activation_circles.extend([node.activation for node in nodes])

    @property
    def patch_ca(self):
        return self._patch_ca

    @patch_ca.setter
    def patch_ca(self, coll):
        self._patch_ca = coll
        self._patch_ca.set_paths(self.activation_circles)

    def draw(self, j, dx, dy):
        for node in self.nodes:
            node.xy = (j * dx, dy * node.y_ord)
        if self.overfull:
            self.overfull_text.set_position((j * dx, 0))

    def set_activation_rad(self, radii):
        for circle, radius in zip(self.activation_circles, radii):
            circle.set_radius(radius)

    def animation_update(self, n_frame):
        normalized_activations = (
            self.visible_activations / self.activations.max()
            if self.normalize_activations
            else self.visible_activations
        )
        factor = n_frame / self.n_frames * Node.radius * 0.9
        self.set_activation_rad(factor * normalized_activations)
        self.patch_ca.set_paths(self.activation_circles)

    def animate_activation(self, **kwargs):
        self.n_frames = kwargs["frames"]
        self.ani = FuncAnimation(**kwargs, func=self.animation_update, repeat=False)

    @property
    def activations(self):
        return self.layer.activation

    @property
    def visible_activations(self):
        return self._visible_activations(self.activations)

    @visible_activations.setter
    def visible_activations(self, n):
        self.n = n
        if n == 0:
            arr = np.empty(self.width)

            def func(A):
                np.divide(A, A.max(), out=arr)
                return arr

        else:
            arr = np.empty(2 * n)

            def func(A):
                np.concat((A[: self.n], A[-self.n :]), out=arr)
                return arr

        self._visible_activations = func

    @property
    def weights(self):
        return self.layer.weights

    @property
    def visible_edge_weights(self):
        return self._visible_edge_weights(self.weights)

    def make_incoming_edges(self, previous):
        n1 = previous.n
        n2 = self.n

        c1 = 2 * n1 or previous.width
        c2 = 2 * n2 or self.width

        size = c1 * c2

        arr = np.empty((c2, c1))
        arr1 = np.empty((self.width, c1))

        if not (n1 or n2):  # both layers are fully visible

            def func(W):
                return W.reshape((size,), copy=False)

        elif (not n1) and n2:  # current layer is not fully visible

            def func(W):
                np.concat((W[:n2, :], W[-n2:, :]), axis=0, out=arr)
                return arr.reshape((size,), copy=False)

        elif n1 and (not n2):  # previous layer is not fully visible

            def func(W):
                np.concat((W[:, :n1], W[:, -n1:]), axis=1, out=arr)
                return arr.reshape((size,), copy=False)

        elif n1 and n2:  # both layers not fully visible

            def func(W):
                np.concat((W[:, :n1], W[:, -n1:]), axis=1, out=arr1)
                np.concat((arr1[:n2, :], arr1[-n2:, :]), axis=0, out=arr)
                return arr.reshape((size,), copy=False)

        self._visible_edge_weights = func
        return size


class MLPVisualization:
    maxwidth = 10

    def __init__(self, mlp_interface, dx, dy, r):

        # create mpl objects
        self.fig = plt.figure()

        # fmt: off
        axs = self.fig.subplot_mosaic(
            [
                ["empty", "empty" ,"box_dxdy", "btn_start"],
                ["img", "graph", "graph", "graph"],   
                ["img", "graph", "graph", "graph"],   
                ["empty", "cbar", "cbar", "cbar"]
            ], 
            width_ratios=[1, 0.4, 0.4, 0.4],
            height_ratios=[
                5,
                45,
                45,
                5
            ],
            empty_sentinel="empty"
        )
        # fmt: on

        self.graph = axs["graph"]
        self.graph.axis("off")
        self.graph.set_aspect("equal")

        self.layers = [LayerVisualization(layer) for layer in mlp_interface.layers]

        self.dx = dx
        self.dy = dy

        Node.radius = r

        for layer in self.layers:
            if layer.width > self.maxwidth:
                layer.overfull = True

        # make nodes
        for layer in self.layers:
            nodes1 = []
            nodes2 = []
            N = layer.width if not layer.overfull else self.maxwidth
            n = int(N / 2)
            centered = N % 2 == 1

            if layer.overfull:
                for k in range(n):
                    y_ord = n - k
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))

            else:
                y_off = 0 if centered else -0.5
                for k in range(n):
                    y_ord = n - k + y_off
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))
                if centered:
                    nodes1.append(Node(i=k + 1, y_ord=0))
                n = 0

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

            layer.visible_activations = n

            coll = PatchCollection(
                [],
                transform=self.graph.transData,
                facecolor="#bdbdbd",
                linewidth=0,
                zorder=2,
            )

            layer.patch_ca = coll
            self.graph.add_collection(coll)

        self.x_minnode = self.layers[0].nodes[0]
        self.x_maxnode = self.layers[-1].nodes[0]
        self.y_maxnode = max(self.layers, key=lambda l: l.width).nodes[0]

        weight_len = 0
        i = 0
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            size = layer2.make_incoming_edges(layer1)
            weight_len += size
            i += 1

        self.weights = np.empty(weight_len)

        self.nodes = [node.node for layer in self.layers for node in layer.nodes]

        self.edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            edges = []
            for node_in in layer2.nodes:
                for node_out in layer1.nodes:
                    edges.append(Edge(node_out, node_in))
            layer2.incoming_edges = edges
            self.edges.extend(edges)

        self.patch_c = PatchCollection(
            [],
            transform=self.graph.transData,
            edgecolor="#bdbdbd",
            facecolor="white",
            linewidth=2 * Node.radius,
            zorder=1,
        )

        self.line_c = LineCollection([], color="#bdbdbd", zorder=-1, linewidth=5)

        for layer in self.layers:
            if layer.overfull:
                layer.overfull_text = self.graph.text(
                    x=0,
                    y=0,
                    s="...",
                    ha="center",
                    va="center",
                    fontsize=24.0 * Node.radius,
                    color="gray",
                    clip_on=True,
                )

        self.graph.add_collection(self.patch_c)
        self.graph.add_collection(self.line_c)

        # make controls
        self.box_dxdy = TextBox(
            ax=axs["box_dxdy"], label="(dx, dy)", textalignment="left"
        )
        self.box_dxdy.on_submit(self.update_geometry)
        self.box_dxdy.set_val(f"{self.dx}, {self.dy}")
        self.btn_start = Button(ax=axs["btn_start"], label="Start")

        self.img = axs["img"].imshow(cmap="gray", vmin=0, vmax=1, X=np.ones((28, 28)))

        # make loss plots
        self.loss_plot = None
        self.accuracy_plot = None
        if 0:
            self.loss_plot = axs["loss_plot"]
            self.train_loss_plot = self.loss_plot.plot(
                [], [], label="Training Loss", color="blue"
            )[0]
            self.val_loss_plot = self.loss_plot.plot(
                [], [], label="Validation Loss", color="red"
            )[0]
            self.loss_plot.legend()
            self.loss_plot.dataLim.x0 = 0
            self.loss_plot.dataLim.y0 = 0

            # make accuracy plots
            self.accuracy_plot = axs["accuracy_plot"]
            self.train_accuracy_plot = self.accuracy_plot.plot(
                [], [], label="Training Accuracy", color="blue"
            )[0]
            self.val_accuracy_plot = self.accuracy_plot.plot(
                [], [], label="Validation Accuracy", color="red"
            )[0]
            self.accuracy_plot.dataLim.x0 = 0
            self.accuracy_plot.dataLim.y0 = 0
            self.accuracy_plot.dataLim.y1 = 1
            self.accuracy_plot.legend()

        # make colorbar
        self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
            cax=axs["cbar"],
            orientation="horizontal",
            label=r"$\text{LogNorm}(W_{ij})$",
        )

        self.norm = SymLogNorm(linthresh=0.03)

        # init weight colors
        self.update_weights()

    def update_weights(self):
        np.concat(
            tuple((layer.visible_edge_weights for layer in self.layers[1:])),
            axis=0,
            out=self.weights,
        )

        self.norm.vmin = self.weights.min()
        self.norm.vmax = self.weights.max()

        self.line_c.set_colors(cmap(self.norm(self.weights)))
        # self.line_c.set_colors(cmap(1 / (1 + np.exp(-self.weights))))

    def update_geometry(self, val):
        dx, dy = self.box_dxdy.text.split(",")
        self.dx = parse_float(dx, default=self.dx)
        self.dy = parse_float(dy, default=self.dy)

        self.draw_graph()
        x_min = self.x_minnode.node.get_extents().xmin
        x_max = self.x_maxnode.node.get_extents().xmax
        y_max = self.y_maxnode.node.get_extents().ymax
        y_min = -y_max
        self.graph.dataLim.x0 = x_min
        self.graph.dataLim.y0 = y_min
        self.graph.dataLim.x1 = x_max
        self.graph.dataLim.y1 = y_max
        self.graph.autoscale_view()
        self.fig.canvas.draw_idle()

    def animate_ff(self):

        frames = 30
        interval = 0.1

        self.ani = FuncAnimation(
            self.fig,
            frames=len(self.layers),
            func=lambda i: self.layers[i].animate_activation(
                fig=self.fig, interval=interval, frames=frames
            ),
            interval=500,
            repeat=False,
        )

        return self.ani

    def draw_graph(self):
        for j, layer in enumerate(self.layers):
            layer.draw(j, self.dx, self.dy)

        self.patch_c.set_paths(self.nodes)
        self.line_c.set_segments([edge.geometry for edge in self.edges])
