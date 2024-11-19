import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.widgets import TextBox
import numpy as np

colors = ["#e6550d", "#fdd0a2", "#000000", "#c7e9c0", "#31a354"]
nodes = [0.0, 0.49, 0.5, 0.51, 1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))


def parse_float(s, default=None):
    try:
        return float(s)
    except ValueError:
        return default


class Node:
    radius = None

    def __init__(self, i, y_ord, xy=(0, 0), **kwargs):
        self.i = i
        self.y_ord = y_ord
        self.xy_ = xy
        self.geometry = (
            Circle(xy=self.xy_, **kwargs)
            if self.radius is None
            else Circle(xy=self.xy_, radius=self.radius, **kwargs)
        )

    @property
    def xy(self):
        return self.xy_

    @xy.setter
    def xy(self, value):
        self.xy_ = value
        self.geometry.center = self.xy_


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class Layer:
    def __init__(self, layer):
        self.layer = layer
        self.width = layer.width

        self.overfull = False
        self.overfull_text = None
        self.count = None

        self.nodes = []
        self.incoming_edges = []

    def add_nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes.extend(nodes)

    def draw(self, j, dx, dy):
        for node in self.nodes:
            node.xy = (j * dx, dy * node.y_ord)
        if self.overfull:
            self.overfull_text.set_position((j * dx, 0))

    @property
    def weights(self):
        if self.incoming_edges:
            if not self.overfull:
                return self.layer.W.T
            else:
                w = self.layer.W.T
                return np.concatenate((w[: self.count + 1], w[-self.count :]))
        return None

    @property
    def activations(self):
        return self.layer.A


class LayerInterface:
    def __init__(self, w):
        self.width = w


class MLPInterface:
    layers = [
        LayerInterface(28 * 28),
        LayerInterface(5),
        LayerInterface(5),
        LayerInterface(10),
    ]


class MLPVisualization:
    maxwidth = 10

    def __init__(self, mlp, dx, dy, r):
        self.layers = [Layer(layer) for layer in mlp.layers]

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

                layer.count = count
            else:
                for k in range(c):
                    y_ord = c - k + y_offset
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))
                if centered:
                    nodes1.append(Node(i=k + 1, y_ord=0))

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

        self.x_minnode = self.layers[0].nodes[0]
        self.x_maxnode = self.layers[-1].nodes[0]
        self.y_maxnode = max(self.layers, key=lambda l: l.width).nodes[0]

        self.patches = [node.geometry for layer in self.layers for node in layer.nodes]

        self.edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            edges = []
            for node_in in layer2.nodes:
                for node_out in layer1.nodes:
                    edges.append(Edge(node_out, node_in))
            layer2.incoming_edges = edges
            self.edges.extend(edges)

        # create mpl objects
        self.fig, ax = plt.subplots(3, 1, height_ratios=[8, 1, 1])
        (self.ax1, self.ax2, self.ax3) = ax
        self.ax1.axis("off")
        self.ax1.set_aspect("equal")

        self.patch_c = PatchCollection(
            [],
            transform=self.ax1.transData,
            edgecolor="black",
            facecolor="white",
            linewidth=2 * Node.radius,
        )
        self.line_c = LineCollection([], color="black", zorder=-1)
        for layer in self.layers:
            if layer.overfull:
                layer.overfull_text = self.ax1.text(
                    x=0,
                    y=0,
                    s="...",
                    ha="center",
                    va="center",
                    fontsize=24.0 * Node.radius,
                )

        self.ax1.add_collection(self.patch_c)
        self.ax1.add_collection(self.line_c)

        self.dx_box = TextBox(ax=self.ax2, label="dx", textalignment="left")
        self.dy_box = TextBox(ax=self.ax3, label="dy", textalignment="left")
        self.dx_box.on_submit(self.update_geometry)
        self.dy_box.on_submit(self.update_geometry)
        self.dx_box.set_val(str(self.dx))
        self.dy_box.set_val(str(self.dy))

        plt.show()

    def update_weights(self):
        weights = np.concatenate(
            (W for layer in self.layers if (W := layer.weights) is not None), axis=None
        )
        w_min = weights.min()
        colors = cmap((weights - w_min) / (weights.max() - w_min))
        self.line_c.set_colors(colors)

    def update_geometry(self, val):
        self.dx = parse_float(self.dx_box.text, default=self.dx)
        self.dy = parse_float(self.dy_box.text, default=self.dy)

        self.draw_nodes()
        x_min = self.x_minnode.geometry.get_extents().xmin
        x_max = self.x_maxnode.geometry.get_extents().xmax
        y_max = self.y_maxnode.geometry.get_extents().ymax
        y_min = -y_max
        self.ax1.dataLim.x0 = x_min
        self.ax1.dataLim.y0 = y_min
        self.ax1.dataLim.x1 = x_max
        self.ax1.dataLim.y1 = y_max
        self.ax1.autoscale_view()
        self.fig.canvas.draw_idle()

    def draw_nodes(self):
        for j, layer in enumerate(self.layers):
            for node in layer.nodes:
                node.xy = (j * self.dx, self.dy * node.y_ord)

        self.patch_c.set_paths(self.patches)
        self.line_c.set_segments([edge.geometry for edge in self.edges])


def main():
    vis = MLPVisualization(mlp=MLPInterface, dx=1, dy=1, r=1)


main()
