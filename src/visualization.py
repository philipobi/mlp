import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    SymLogNorm,
    ListedColormap,
)
from matplotlib.widgets import TextBox, Button
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
from itertools import cycle, zip_longest
from utils import Iterator, flatten, cmap_red_green, Array

PatchCollection.update_objects = lambda self: self.set_paths(self.objects)

transitionLinear = lambda i: i


class SimplePlot:
    def __init__(
        self,
        ax,
        xlims,
        ylims,
        n_plots=1,
        datalen=10000,
        x_increment=100,
        plot_kwargs=(),
    ):
        self.ax = ax
        self.ax.grid()
        _, self.xmax = xlims
        self.ax.set_xlim(*xlims)
        self.ax.set_ylim(*ylims)
        self.x_increment = x_increment
        self.arr = Array(shape=(datalen, n_plots + 1))

        self.plots = [
            self.ax.plot([], [], **kwargs)[0]
            for _, kwargs in zip_longest(range(n_plots), plot_kwargs, fillvalue={})
        ]

    def update(self, x, *args):
        self.arr.insert([x, *args])

        if x >= self.xmax:
            self.xmax += self.x_increment
            self.ax.set_xlim(None, self.xmax)

        data = self.arr.data.T
        x = data[0]
        for i, plot in enumerate(self.plots, start=1):
            plot.set_xdata(x)
            plot.set_ydata(data[i])


class AnimationSpec:
    def __init__(self, func, transition, frames):
        self.func = func
        self.transition = transition
        self.frames = frames


class AnimationIterator(Iterator):
    def __init__(self, spec_it):
        self.ani = iter(spec_it)
        self.spec = None
        self.completed = False

    def __next__(self):
        if self.spec is None:
            self.spec = next(self.ani)
            self.it = iter(range(1, self.spec.frames + 1))
            return next(self)
        else:
            try:
                i = self.spec.transition(next(self.it) / self.spec.frames)
                self.spec.func(i)
                return None
            except StopIteration:
                self.spec = None
                return next(self)


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
    dx = None
    dy = None


class Node:
    radius = None
    ctx = Context

    def __init__(self, y_ord, xy=(0, 0), **kwargs):
        self.parent = None
        self.label = None
        self.y_ord = y_ord
        self.xy_ = xy

        self.node = Circle(xy=self.xy_, radius=Node.radius, **kwargs)
        self.activation = Circle(xy=self.xy_, radius=0, **kwargs)

    def add_label(self, text):
        x, y = self.xy
        self.label = self.ctx.graph.text(
            s=str(text), x=x + 2 * Node.radius, y=y, zorder=3, clip_on=True, va="center"
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
            self.label.set_position(xy=(x + 1.5 * Node.radius, y))


class OverfullIndicator:
    def __init__(self):
        self.node = Circle(xy=(0, 0), radius=Node.radius)
        self.circles = [Circle(xy=(0, 0), radius=Node.radius / 8) for _ in range(3)]

    @property
    def xy(self):
        return self.node.center

    @xy.setter
    def xy(self, xy):
        x, y = xy
        self.node.center = (x, y)
        for i, circle in enumerate(self.circles, start=-1):
            circle.center = (x + i * Node.radius / 2, y)


class Edge:
    ctx = Context

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class LayerVisualization:
    ctx = Context

    def __init__(self, layer, i):
        self.previous = None
        self.i = i
        self.layer = layer
        self.width = layer.width

        self.normalize_activations = True

        self.n = None
        self.m = None
        self.W_size = 0

        self._overfull = False
        self.overfull_ind = None
        self.annotated = False

        self.nodes = []
        self.bbox = self.ctx.graph.add_patch(
            Rectangle(
                xy=(0, 0),
                width=0,
                height=0,
                facecolor="none",
                edgecolor="#bdbdbd",
                alpha=0.0,
            )
        )

        self.node_activations = PatchCollection(
            [],
            transform=self.ctx.graph.transData,
            facecolor="#bdbdbd",
            linewidth=0,
            zorder=2,
        )
        self.node_activations.objects = []

        self._visible_edge_weights = lambda W: None
        self._visible_activations = lambda A: None

    def add_nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes.extend(nodes)
        self.node_activations.objects.extend([node.activation for node in nodes])

    def annotate(self, labels):
        self.annotated = True
        for node, label in zip(self.nodes, labels):
            node.add_label(label)

    def draw(self):
        for node in self.nodes:
            node.xy = (self.i * self.ctx.dx, self.ctx.dy * node.y_ord)

        r = Node.radius
        pad = r / 2
        x0, y0 = self.nodes[-1].xy
        x1, y1 = self.nodes[0].xy
        self.bbox.set_xy((x0 - r - pad, y0 - r - pad))
        self.bbox.set_height(y1 - y0 + 2 * r + 2 * pad)
        self.bbox.set_width(x1 - x0 + 2 * r + 2 * pad)

        self.node_activations.update_objects()

        if self.overfull:
            self.overfull_ind.xy = (self.i * self.ctx.dx, 0)

    @property
    def overfull(self):
        return self._overfull

    @overfull.setter
    def overfull(self, value: bool):
        self._overfull = value
        if self.overfull and not self.overfull_ind:
            self.overfull_ind = OverfullIndicator()

    def set_activation_rad(self, radii):
        for circle, radius in zip(self.node_activations.objects, radii):
            circle.set_radius(radius)

    def reset_activation_rad(self):
        for circle in self.node_activations.objects:
            circle.set_radius(0)
        self.node_activations.update_objects()

    def animate_activation(self, frames):
        arr = np.empty_like(self.visible_activations)
        arr1 = np.empty_like(self.visible_activations)
        (
            np.divide(self.visible_activations, self.activations.max(), out=arr)
            if self.normalize_activations
            else np.copyto(src=self.activations, dst=arr)
        )

        def func(i):
            np.multiply(arr, i * Node.radius * 0.9, out=arr1)
            self.set_activation_rad(arr1)
            self.node_activations.update_objects()

        yield from [
            AnimationSpec(
                func=lambda i: self.bbox.set_alpha(i),
                transition=transitionLinear,
                frames=20,
            ),
            AnimationSpec(func=func, transition=transitionLinear, frames=15),
            AnimationSpec(
                func=lambda i: self.bbox.set_alpha(1 - i),
                transition=transitionLinear,
                frames=20,
            ),
        ]

    @property
    def activations(self):
        return self.layer.activation

    @property
    def visible_activations(self):
        return self._visible_activations(self.activations)

    def set_visible_nodes(self, n, m):
        self.n = n
        self.m = m

        if n == m:

            def func_A(A):
                return A

        else:

            def func_A(A):
                return A[n:m]

        if self.previous:
            n0, m0 = (self.previous.n, self.previous.m)
            i, j = (
                (self.width if n == m else m - n),
                (self.previous.width if n0 == m0 else m0 - n0),
            )

            W_slice = np.empty((i, j))
            self.W_size = i * j

            if n == m and n0 == m0:

                def func_W(W):
                    np.copyto(src=W, dst=W_slice)
                    return np.reshape(W_slice, (-1,), copy=False)

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

    def __init__(self, layers_interfaces, dx, dy, r):

        # create mpl objects
        self.fig = plt.figure()

        """
        accuracy    controls (padding, btn_cmap, box_dxdy, btn_start)
        accuracy    empty   graph   empty
        3d          img     graph   cbar
        3d          img     graph   cbar
        3d          empty   graph   empty
        """

        gs = GridSpec(nrows=5, ncols=4)
        gs.set_height_ratios((5, 25, 20, 25, 25))
        gs.set_width_ratios((40, 10, 45, 5))

        gs_controls = gs[0, 1:].subgridspec(nrows=1, ncols=4)
        ax_btn_cmap = self.fig.add_subplot(gs_controls[0, -3])
        ax_box_dxdy = self.fig.add_subplot(gs_controls[0, -2])
        ax_btn_start = self.fig.add_subplot(gs_controls[0, -1])
        gs_controls.set_width_ratios((70, 10, 10, 10))

        ax_accuracy = self.fig.add_subplot(gs[:2, 0])
        ax_loss_projection = self.fig.add_subplot(gs[2:, 0], projection="3d")
        ax_img = self.fig.add_subplot(gs[2:4, 1])
        ax_graph = self.fig.add_subplot(gs[1:, 2])
        ax_cbar = self.fig.add_subplot(gs[2:4, 3])

        gs.tight_layout(self.fig)

        self.graph = ax_graph
        self.graph.axis("off")
        self.graph.set_aspect("equal")

        Context.fig = self.fig
        Context.graph = self.graph
        Context.dx = dx
        Context.dy = dy

        self.layers = [
            LayerVisualization(layer, i) for i, layer in enumerate(layers_interfaces)
        ]

        Node.radius = r

        for layer in self.layers:
            if layer.width > self.maxwidth:
                layer.overfull = True

        # make nodes
        previous = None
        for layer in self.layers:
            layer.previous = previous
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
                layer.set_visible_nodes(n=w - n, m=w + n - 1)

            else:
                centered = N % 2 == 1
                y_off = 0 if centered else -0.5
                for k in range(n):
                    y_ord = n - k + y_off
                    nodes1.append(Node(y_ord))
                    nodes2.append(Node(-y_ord))
                if centered:
                    nodes1.append(Node(y_ord=0))
                layer.set_visible_nodes(n=0, m=0)

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

            previous = layer

        bbox = max(self.layers, key=lambda l: l.width).bbox
        self.ylim = lambda: (y := bbox.get_y() and None) or (y, y + bbox.get_height())

        self.x_minnode = self.layers[0].nodes[0]
        self.x_maxnode = self.layers[-1].nodes[0]
        self.y_maxlayer = max(self.layers, key=lambda l: l.width).nodes[0]

        self.weights = np.empty(sum([layer.W_size for layer in self.layers]))

        self.nodes = PatchCollection(
            [],
            transform=self.graph.transData,
            edgecolor="#bdbdbd",
            facecolor="white",
            linewidth=Node.radius,
            zorder=1,
        )

        nodes = [node.node for layer in self.layers for node in layer.nodes]
        nodes.extend(
            [layer.overfull_ind.node for layer in self.layers if layer.overfull]
        )
        self.nodes.objects = nodes

        self.overfull_dots = PatchCollection(
            [], zorder=2, facecolor="#bdbdbd", edgecolor="none"
        )
        overfull_dots = [
            circle
            for layer in self.layers
            if layer.overfull
            for circle in layer.overfull_ind.circles
        ]
        self.overfull_dots.objects = overfull_dots

        self.edges = LineCollection([], color="#bdbdbd", zorder=0, linewidth=2)
        edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            _edges = []
            for node_in in layer2.nodes:
                for node_out in layer1.nodes:
                    _edges.append(Edge(node_out, node_in))
            edges.extend(_edges)
        self.edges.objects = edges

        self.graph.add_collection(self.nodes)
        self.graph.add_collection(self.edges)
        self.graph.add_collection(self.overfull_dots)
        for layer in self.layers:
            self.graph.add_collection(layer.node_activations)

        # make controls
        self.box_dxdy = TextBox(ax=ax_box_dxdy, label="(dx, dy)", textalignment="left")
        self.box_dxdy.on_submit(self.update_geometry)
        self.box_dxdy.set_val(f"{Context.dx}, {Context.dy}")
        self.btn_start = Button(ax=ax_btn_start, label="Start/Pause")

        self.ax_img = ax_img
        self.ax_img.axis("off")

        self.ax_loss_projection = ax_loss_projection

        self.ax_accuracy = ax_accuracy

        # make loss plots
        self.loss_plot = None
        self.accuracy_plot = None
        if 1:
            # make accuracy plots
            self.accuracy_plot = ax_accuracy
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
        cmap.set(
            mpl.colormaps["tab20c"],
            cmap_red_green,
        )

        cmap.norm = SymLogNorm(linthresh=0.03)
        self.cbar_ax = ax_cbar
        self.make_cbar()

        # add button for switching cbar
        self.btn_cmap = Button(ax=ax_btn_cmap, label="Switch cmap")
        self.btn_cmap.on_clicked(self.switch_cmap)

        # init weight colors
        self.update_weights()

    @property
    def bbox(self):
        y0, y1 = self.ylim()

    def make_cbar(self):
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap.cmap),
            cax=self.cbar_ax,
            location="left",
            orientation="vertical",
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

        self.edges.set_colors(cmap.map(self.weights))
        # self.line_c.set_colors(cmap.cmap(1 / (1 + np.exp(-self.weights))))

    def switch_cmap(self, _=None):
        cmap.switch()
        self.make_cbar()
        self.update_weights()

    def update_geometry(self, _):
        dx, dy = self.box_dxdy.text.split(",")
        Context.dx = parse_float(dx, default=Context.dx)
        Context.dy = parse_float(dy, default=Context.dy)

        self.draw_graph()

        self.fig.draw_without_rendering()

        bbox = max(self.layers, key=lambda l: l.width).bbox
        _, y0 = bbox.xy
        y1 = y0 + bbox.get_height()
        x0, _ = self.layers[0].bbox.xy

        layer = self.layers[-1]
        if layer.annotated:
            inv = self.graph.transData.inverted()
            x1 = max(
                map(
                    lambda node: inv.transform(node.label.get_window_extent())[1, 0],
                    layer.nodes,
                )
            )
        else:
            x, _ = layer.bbox.xy
            x1 = x + layer.bbox.get_width()

        self.graph.dataLim.x0 = x0
        self.graph.dataLim.y0 = y0
        self.graph.dataLim.x1 = x1
        self.graph.dataLim.y1 = y1
        self.graph.autoscale_view()
        self.fig.canvas.draw_idle()

    def animate_ff(self):

        yield from AnimationIterator(
            flatten(
                map(lambda layer: layer.animate_activation(frames=30), self.layers),
                dim=2,
            )
        )

        # return frames

    def clear_activations(self):
        for layer in self.layers:
            layer.reset_activation_rad()

    def draw_graph(self):
        for layer in self.layers:
            layer.draw()

        self.nodes.update_objects()

        self.edges.set_segments([edge.geometry for edge in self.edges.objects])

        self.overfull_dots.update_objects()
