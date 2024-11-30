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
cmap = mpl.colormaps["tab20c"]


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

        self.overfull = False
        self.overfull_text = None
        self.count = None

        self.nodes = []
        self.incoming_edges = []

        self._get_edge_weights = lambda W: None
        self._get_activations = lambda A: None

    def add_nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes.extend(nodes)

    def draw(self, j, dx, dy):
        for node in self.nodes:
            node.xy = (j * dx, dy * node.y_ord)
        if self.overfull:
            self.overfull_text.set_position((j * dx, 0))

    def get_edge_weights(self):
        if self.incoming_edges:
            return self._get_edge_weights(self.weights)

    def get_activations(self):
        return self._get_activations(self.activations)

    def set_activation_rad(self, rad):
        for node, radius in zip(self.nodes, rad):
            node.activation.set_radius(radius)

    @property
    def activations(self):
        return self.layer.activations

    @property
    def weights(self):
        return self.layer.weights


def edge_weight_func(c1, w1, c2, w2):
    c1_2 = int(c1 / 2)
    c2_2 = int(c2 / 2)

    size = c1 * c2

    arr = np.empty((c2, c1))
    arr1 = np.empty((w2, c1))

    if c1 == w1 and c2 == w2:

        def func(W):
            return W.reshape((size,), copy=False)

    elif c1 == w1 and c2 != w2:

        def func(W):
            np.concat((W[:c2_2, :], W[-c2_2:, :]), axis=0, out=arr)
            return arr.reshape((size,), copy=False)

    elif c1 != w1 and c2 == w2:

        def func(W):
            np.concat((W[:, :c1_2], W[:, -c1_2:]), axis=1, out=arr)
            return arr.reshape((size,), copy=False)

    elif c1 != w1 and c2 != w2:

        def func(W):
            np.concat((W[:, :c1_2], W[:, -c1_2:]), axis=1, out=arr1)
            np.concat((arr1[:c2_2, :], arr1[-c2_2:, :]), axis=0, out=arr)
            return arr.reshape((size,), copy=False)

    return (size, func)


def activation_func(c, w):
    arr = np.empty(c)
    c_2 = int(c / 2)
    if c == w:

        def func(A):
            return A

    if c != w:

        def func(A):
            np.concat(A[:c_2], A[-c_2:], axis=0, out=arr)
            return arr

    return func


class MLPVisualization:
    maxwidth = 10

    def __init__(self, mlp_interface, dx, dy, r):

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
            centered = N % 2 == 1
            c = int(N / 2)
            y_offset = 0 if centered else -0.5

            if layer.overfull:
                for k in range(count := (c if centered else c - 1)):
                    y_ord = c - k + y_offset
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))

                layer.count = 2 * count

            else:
                for k in range(count := c):
                    y_ord = c - k + y_offset
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))
                if centered:
                    nodes1.append(Node(i=k + 1, y_ord=0))

                layer.count = layer.width

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

            layer._get_activations = activation_func(count, layer.width)

        self.x_minnode = self.layers[0].nodes[0]
        self.x_maxnode = self.layers[-1].nodes[0]
        self.y_maxnode = max(self.layers, key=lambda l: l.width).nodes[0]

        weight_len = 0
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            (size, layer2._get_edge_weights) = edge_weight_func(
                layer1.count, layer1.width, layer2.count, layer2.width
            )
            weight_len += size

        self.weights = np.empty(weight_len)

        self.nodes = [node.node for layer in self.layers for node in layer.nodes]
        self.activations = [
            node.activation for layer in self.layers for node in layer.nodes
        ]

        self.edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            edges = []
            for node_in in layer2.nodes:
                for node_out in layer1.nodes:
                    edges.append(Edge(node_out, node_in))
            layer2.incoming_edges = edges
            self.edges.extend(edges)

        # create mpl objects
        self.fig = plt.figure()

        # fmt: off
        axs = self.fig.subplot_mosaic(
            [
                ["loss_plot", "empty" ,"box_dxdy", "btn_start"],
                ["loss_plot", "graph", "graph", "graph"],   
                ["accuracy_plot", "graph", "graph", "graph"],   
                ["accuracy_plot", "cbar", "cbar", "cbar"]
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

        self.patch_c = PatchCollection(
            [],
            transform=self.graph.transData,
            edgecolor="#bdbdbd",
            facecolor="white",
            linewidth=2 * Node.radius,
            zorder=1,
        )

        self.patch_ca = PatchCollection(
            [],
            transform=self.graph.transData,
            facecolor="#bdbdbd",
            linewidth=0,
            zorder=2,
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
        self.graph.add_collection(self.patch_ca)
        self.graph.add_collection(self.line_c)

        # make controls
        self.box_dxdy = TextBox(
            ax=axs["box_dxdy"], label="(dx, dy)", textalignment="left"
        )
        self.box_dxdy.on_submit(self.update_geometry)
        self.box_dxdy.set_val(f"{self.dx}, {self.dy}")
        self.btn_start = Button(ax=axs["btn_start"], label="Start")

        # make loss plots
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
            tuple((layer.get_edge_weights() for layer in self.layers[1:])),
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

    def update_activations(self):
        self.patch_ca.set_paths(self.activations)

    def draw_graph(self):
        for j, layer in enumerate(self.layers):
            layer.draw(j, self.dx, self.dy)

        self.patch_c.set_paths(self.nodes)
        self.patch_ca.set_paths(self.activations)
        self.line_c.set_segments([edge.geometry for edge in self.edges])
