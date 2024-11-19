import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
from matplotlib.widgets import TextBox, Button
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import numpy as np

colors = ["#e6550d", "#fdd0a2", "#bdbdbd", "#c7e9c0", "#31a354"]
nodes = [0.0, 0.49, 0.5, 0.51, 1.0]
cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
cmap = mpl.colormaps["tab20c"]


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
        self.weight = 0

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class LayerVisualization:
    def default_callback(self, event):
        pass

    def __init__(self, layer, callbacks={}):
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

    def update_edge_weights(self):
        if self.incoming_edges:
            W = self.weights
            for edge in self.incoming_edges:
                edge.weight = W[edge.node2.i, edge.node1.i]

    @property
    def activations(self):
        return self.layer.activations

    @property
    def weights(self):
        return self.layer.weights


class MLPVisualization:
    training_interval = 20
    maxwidth = 10
    default_hooks = {"train_batch": lambda: False, "stop_training": lambda: None}

    def __init__(self, mlp_interface, dx, dy, r):
        self._train_batch = mlp_interface.hooks.get(
            "train_batch", self.default_hooks["train_batch"]
        )
        self._stop_training = mlp_interface.hooks.get(
            "stop_training", self.default_hooks["stop_training"]
        )

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
        self.fig = plt.figure(layout="constrained")

        # fmt: off
        axs = self.fig.subplot_mosaic(
            [
                ["box_dxdy", "graph"], 
                ["btn_start", "graph"], 
                ["btn_stop", "graph"],
                ["batch_c", "graph"],
                ["empty", "graph"]
            ], 
            height_ratios=[5, 5, 5, 5, 80],
            width_ratios=[5, 95],
            empty_sentinel="empty"
        )
        # fmt: on

        self.ax1 = axs["graph"]
        self.ax1.axis("off")
        self.ax1.set_aspect("equal")

        self.patch_c = PatchCollection(
            [],
            transform=self.ax1.transData,
            edgecolor="#bdbdbd",
            facecolor="white",
            linewidth=2 * Node.radius,
        )
        self.line_c = LineCollection([], color="#bdbdbd", zorder=-1, linewidth=5)

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

        self.box_dxdy = TextBox(
            ax=axs["box_dxdy"], label="(dx, dy)", textalignment="left"
        )
        self.box_dxdy.on_submit(self.update_geometry)
        self.box_dxdy.set_val(f"{self.dx}, {self.dy}")

        btn_start = Button(ax=axs["btn_start"], label="Start")
        self.ani = None
        btn_start.on_clicked(self.start_training)
        btn_stop = Button(ax=axs["btn_stop"], label="Stop")
        btn_stop.on_clicked(self.stop_training)

        batchc_ax = axs["batch_c"]
        self.batchc_text = batchc_ax.text(
            x=0,
            y=0,
            s=f"Batches: {0:06}",
            ha="center",
            va="center",
            fontsize=12.0,
        )
        batchc_ax.axis("off")
        batchc_ax.autoscale(enable=True)

        self.norm = SymLogNorm(linthresh=0.03)

        self.update_weights()
        plt.show()

    def update_frame(self, frame):
        if not self._train_batch():
            self.ani.event_source.stop()
        else:
            self.update_weights()
            self.batchc_text.set_text(f"Batches: {frame+1:06}")

    def start_training(self, _):
        self.ani = FuncAnimation(
            fig=self.fig,
            func=self.update_frame,
            frames=None,
            cache_frame_data=False,
            interval=self.training_interval,
        )

    def stop_training(self, _):
        if self.ani is not None:
            self.ani.event_source.stop()
            print("stopped animation")

    def update_weights(self):
        for layer in self.layers:
            layer.update_edge_weights()

        weights = np.array([edge.weight for edge in self.edges])

        self.norm.vmin = weights.min()
        self.norm.vmax = weights.max()

        self.line_c.set_colors(cmap(self.norm(weights)))

    def update_geometry(self, val):
        dx, dy = self.box_dxdy.text.split(",")
        self.dx = parse_float(dx, default=self.dx)
        self.dy = parse_float(dy, default=self.dy)

        self.draw_graph()
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

    def draw_graph(self):
        for j, layer in enumerate(self.layers):
            layer.draw(j, self.dx, self.dy)

        self.patch_c.set_paths(self.patches)
        self.line_c.set_segments([edge.geometry for edge in self.edges])
