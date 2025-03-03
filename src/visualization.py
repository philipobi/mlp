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
from itertools import cycle

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


class cmap:
    _cmaps = None
    cmap = None
    norm = None

    @staticmethod
    def map(arr):
        return cmap.cmap(cmap.norm(arr))

    @staticmethod
    def switch():
        cmap.cmap = next(cmap._cmaps)

    @staticmethod
    def set(*cmaps):
        cmap._cmaps = cycle(cmaps)
        cmap.cmap = next(cmap._cmaps)


class Context:
    fig = None
    graph = None


class Node:
    radius = None
    ctx = Context

    def __init__(self, y_ord, xy=(0, 0), **kwargs):
        self.label = None
        self.y_ord = y_ord
        self.xy_ = xy

        if kwargs.get("radius", None) is None:
            kwargs["radius"] = self.radius

        self.node = Circle(xy=self.xy_, **kwargs)
        kwargs["radius"] = 0.0
        self.activation = Circle(xy=self.xy_, **kwargs)

    def add_label(self, text):
        x, y = self.xy
        self.label = self.ctx.graph.text(
            s=text, x=x + 1.5 * self.radius, y=y, zorder=3, clip_on=True, va="center"
        )

    @property
    def xy(self):
        return self.xy_

    @xy.setter
    def xy(self, value):
        self.xy_ = value
        self.node.center = self.xy_
        self.activation.center = self.xy_
        if self.label:
            x, y = value
            self.label.set_position(xy=(x + 1.5 * self.radius, y))


class Edge:
    ctx = Context

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.weight = 0

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class LayerVisualization:
    ctx = Context

    def default_callback(self, event):
        pass

    def __init__(self, layer):
        self.layer = layer
        self.width = layer.width

        self.n = None
        self.m = None
        self.W_size = 0

        self._overfull = False
        self.overfull_text = None

        self.nodes = []
        self.incoming_edges = []
        self.activation_circles = []
        self._patch_ca = None

        self._visible_edge_weights = lambda W: None
        self._visible_activations = lambda A: None
        self.normalize_activations = True

    def add_nodes(self, nodes):
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

    @property
    def overfull(self):
        return self._overfull

    @overfull.setter
    def overfull(self, value: bool):
        self._overfull = value
        if self.overfull and not self.overfull_text:
            self.overfull_text = self.ctx.graph.text(
                x=0,
                y=0,
                s="...",
                ha="center",
                va="center",
                fontsize=24.0 * Node.radius,
                color="gray",
                clip_on=True,
            )

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

    def reset_activation_rad(self):
        for circle in self.activation_circles:
            circle.set_radius(0)
        self.patch_ca.set_paths(self.activation_circles)

    @property
    def activations(self):
        return self.layer.activation

    @property
    def visible_activations(self):
        return self._visible_activations(self.activations)

    def set_visible_nodes(self, n, m, previous):
        self.n = n
        self.m = m

        if n == m:
            A_normal = np.empty(self.width)

            def func_A(A):
                np.divide(A, A.max(), out=A_normal)
                return A_normal

        else:
            A_normal = np.empty(m - n)

            def func_A(A):
                np.divide(A[n:m], A.max(), out=A_normal)
                return A_normal

        if previous:
            n0, m0 = (previous.n, previous.m)
            i, j = (
                (self.width if n == m else m - n),
                (previous.width if n0 == m0 else m0 - n0),
            )

            W_slice = np.empty((i, j))
            self.W_size = i * j

            if n == m and n0 == m0:

                def func_W(W):
                    return np.reshape(W, (-1,), copy=False)

            elif n == m and n0 != m0:

                def func_W(W):
                    np.copyto(src=W[:, n0:m0], dst=W_slice)
                    return np.reshape(W_slice, (-1,), copy=False)

            elif n != m and n0 == m0:

                def func_W(W):
                    np.copyto(src=W[n:m, :], dst=W_slice)
                    return np.reshape(W_slice, (-1,), copy=False)

            else:

                def func_W(W):
                    np.copyto(src=W[n:m, n0:m0], dst=W_slice)
                    return np.reshape(W_slice, (-1,), copy=False)

            self._visible_edge_weights = func_W

        self._visible_activations = func_A

    @property
    def weights(self):
        return self.layer.weights

    @property
    def visible_edge_weights(self):
        return self._visible_edge_weights(self.weights)


class MLPVisualization:
    maxwidth = 10

    def __init__(self, mlp_interface, dx, dy, r):

        # create mpl objects
        self.fig = plt.figure()

        # fmt: off
        axs = self.fig.subplot_mosaic(
            [
                ["empty", "btn_cmap" ,"box_dxdy", "btn_start"],
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

        Context.fig = self.fig
        Context.graph = self.graph

        self.layers = [LayerVisualization(layer) for layer in mlp_interface.layers]

        self.dx = dx
        self.dy = dy

        Node.radius = r

        for layer in self.layers:
            if layer.width > self.maxwidth:
                layer.overfull = True

        # make nodes
        previous = None
        for layer in self.layers:
            nodes1 = []
            nodes2 = []
            N = layer.width if not layer.overfull else self.maxwidth
            n = int(N / 2)

            if layer.overfull:
                for k in range(n):
                    y_ord = n - k
                    nodes1.append(Node(y_ord))
                    nodes2.append(Node(-y_ord))

                w = int(layer.width / 2)
                layer.set_visible_nodes(n=w - n, m=w + n - 1, previous=previous)

            else:
                centered = N % 2 == 1
                y_off = 0 if centered else -0.5
                for k in range(n):
                    y_ord = n - k + y_off
                    nodes1.append(Node(y_ord))
                    nodes2.append(Node(-y_ord))
                if centered:
                    nodes1.append(Node(y_ord=0))
                layer.set_visible_nodes(n=0, m=0, previous=previous)

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

            coll = PatchCollection(
                [],
                transform=self.graph.transData,
                facecolor="#bdbdbd",
                linewidth=0,
                zorder=2,
            )

            layer.patch_ca = coll
            self.graph.add_collection(coll)

            previous = layer

        self.x_minnode = self.layers[0].nodes[0]
        self.x_maxnode = self.layers[-1].nodes[0]
        self.y_maxnode = max(self.layers, key=lambda l: l.width).nodes[0]

        self.weights = np.empty(sum([layer.W_size for layer in self.layers]))

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
            linewidth=Node.radius,
            zorder=1,
        )

        self.line_c = LineCollection([], color="#bdbdbd", zorder=-1, linewidth=2)

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

        # setup cmap and cbar
        cmap.set(mpl.colormaps["viridis"], ListedColormap(colors=colors))
        cmap.norm = SymLogNorm(linthresh=0.03)
        self.cbar_ax = axs["cbar"]
        self.make_cbar()
        
        #add button for switching cbar
        self.btn_cmap = Button(ax=axs["btn_cmap"], label="Switch cmap")
        self.btn_cmap.on_clicked(self.switch_cmap)

        # init weight colors
        self.update_weights()

    def make_cbar(self):
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap.cmap),
            cax=self.cbar_ax,
            orientation="horizontal",
            label=r"$\text{LogNorm}(W_{ij})$",
        )

    def update_weights(self):
        np.concat(
            tuple((layer.visible_edge_weights for layer in self.layers[1:])),
            axis=0,
            out=self.weights,
        )

        cmap.norm.vmin = self.weights.min()
        cmap.norm.vmax = self.weights.max()

        self.line_c.set_colors(cmap.map(self.weights))
        # self.line_c.set_colors(cmap.cmap(1 / (1 + np.exp(-self.weights))))

    def switch_cmap(self, _=None):
        cmap.switch()
        self.make_cbar()
        self.update_weights()

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

    def clear_activations(self):
        for layer in self.layers:
            layer.reset_activation_rad()

    def draw_graph(self):
        for j, layer in enumerate(self.layers):
            layer.draw(j, self.dx, self.dy)

        self.patch_c.set_paths(self.nodes)
        self.line_c.set_segments([edge.geometry for edge in self.edges])
