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


colors = [
    "#e6550d",
    "#fd8d3c",
    "#fdae6b",
    "#fdd0a2",
    "#d9d9d9",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#31a354",
]

# palette: https://medium.com/design-bootcamp/creating-a-consistent-color-palette-for-your-interface-870e47a4206a
# dark -> light
red = [
    "#89012A",
    "#A0021F",
    "#B40317",
    "#C20812",
    "#D00C0C",
    "#E23939",
    "#F46868",
]
# light -> dark
green = [
    "#67D451",
    "#3BBD2E",
    "#0FA50C",
    "#0C9012",
    "#0D7E1C",
    "#086820",
    "#025322",
]


colors = [
    (0.90196078, 0.33333333, 0.05098039),
    (0.9290092, 0.39919036, 0.10625325),
    (0.95605762, 0.46504738, 0.16152611),
    (0.98518669, 0.53597033, 0.22105073),
    (0.99215686, 0.5817491, 0.27632359),
    (0.99215686, 0.62354298, 0.3358482),
    (0.99215686, 0.66235158, 0.39112106),
    (0.99215686, 0.70635919, 0.45844148),
    (0.99215686, 0.74911004, 0.52759727),
    (0.99215686, 0.79514942, 0.60207274),
    (0.75879737, 0.90463295, 0.73191467),
    (0.71038631, 0.88424934, 0.68477759),
    (0.66543319, 0.86532171, 0.64100744),
    (0.61447859, 0.84309655, 0.59395255),
    (0.56155777, 0.81840016, 0.55043988),
    (0.50456612, 0.79180406, 0.50358007),
    (0.4497501, 0.76608997, 0.46013072),
    (0.35959246, 0.72168397, 0.41437908),
    (0.27587466, 0.68044983, 0.37189542),
    (0.19215686, 0.63921569, 0.32941176),
]


# colors = ["#e6550d", "#fdd0a2", "#bdbdbd", "#c7e9c0", "#31a354"]
# nodes = [0.0, 0.17, 0.34, 0.499, 0.5, 0.501, 0.67, 0.84, 1.0]
# cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

cmap = ListedColormap(colors=colors)


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

        self.patches = [node.node for layer in self.layers for node in layer.nodes]

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

        # self.line_c.set_colors(cmap(self.norm(self.weights)))
        self.line_c.set_colors(cmap(1 / (1 + np.exp(-self.weights))))

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

    def draw_graph(self):
        for j, layer in enumerate(self.layers):
            layer.draw(j, self.dx, self.dy)

        self.patch_c.set_paths(self.patches)
        self.line_c.set_segments([edge.geometry for edge in self.edges])
