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

transitionLinear = lambda x: x
PatchCollection.update_objects = lambda self: self.set_paths(self.objects)


class WindowedPlot:
    def __init__(self, ax, ylims, n_plots=1, windowsize=200, plot_kwargs=()):
        self.ax = ax
        self.ax.grid()
        self.windowsize = windowsize
        self.xmin = 0
        self.xmax = self.windowsize
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(*ylims)
        self.plots = [
            self.ax.plot([], [], **kwargs)[0]
            for _, kwargs in zip_longest(range(n_plots), plot_kwargs, fillvalue={})
        ]
        self.arr = Array(shape=(self.windowsize, n_plots + 1))
        self.i = 0

    def move_window(self):
        (self.xmin, self.xmax) = (self.xmax, self.xmax + self.windowsize)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.arr.clear()

    def update(self, x, *args):
        if x >= self.xmax:
            self.move_window()
        self.arr.insert([x, *args])

        data = self.arr.data.T
        x = data[0]
        for i, plot in enumerate(self.plots, start = 1):
            plot.set_xdata(x)
            plot.set_ydata(data[i])


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


class cmap:
    def __init__(
        self, cmap, norm, compute_limits=lambda values: (values.min(), values.max())
    ):
        self.cmap = cmap
        self.norm = norm
        self.compute_limits = compute_limits

    def map(self, values):
        (self.norm.vmin, self.norm.vmax) = self.compute_limits(values)
        return self.cmap(self.norm(values))


class Context:
    fig = None
    graph = None
    dx = None
    dy = None


class Node:
    radius = None
    ctx = Context

    def __init__(self, ord, **kwargs):
        self.parent = None
        x_ord, y_ord = ord
        self.xy = (x_ord * self.ctx.dx, y_ord * self.ctx.dy)

        self.node = Circle(xy=self.xy, radius=Node.radius, **kwargs)
        self.activation = Circle(xy=self.xy, radius=0, **kwargs)


class OverfullIndicator:
    ctx = Context

    def __init__(self, ord):
        x_ord, y_ord = ord
        x = x_ord * self.ctx.dx
        y = y_ord * self.ctx.dy
        r = Node.radius

        self.node = Circle(xy=(x, y), radius=r)
        self.circles = [
            Circle(xy=(x + i * r / 2, y), radius=r / 8) for i in range(-1, 2)
        ]


class Edge:
    ctx = Context

    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.geometry = [self.node1.xy, self.node2.xy]


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

        self.nodes_ = []

        self.node_activations = PatchCollection(
            [],
            transform=self.ctx.graph.transData,
            facecolor="#bdbdbd",
            linewidth=0,
            zorder=2,
        )

        self._visible_edge_weights = lambda W: None
        self._visible_activations = lambda A: None

    @property
    def nodes(self):
        return self.nodes_

    @nodes.setter
    def nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes_ = nodes
        self.node_activations.objects = [node.activation for node in nodes]

        r = Node.radius
        pad = r / 2
        x0, y0 = self.nodes[-1].xy
        x1, y1 = self.nodes[0].xy
        xy = (x0 - r - pad, y0 - r - pad)
        height = y1 - y0 + 2 * r + 2 * pad
        width = x1 - x0 + 2 * r + 2 * pad
        self.bbox = self.ctx.graph.add_patch(
            Rectangle(
                xy=xy,
                width=width,
                height=height,
                facecolor="none",
                edgecolor="#bdbdbd",
                alpha=0.0,
            )
        )

    @property
    def overfull(self):
        return self._overfull

    @overfull.setter
    def overfull(self, value: bool):
        self._overfull = value
        if self.overfull and not self.overfull_ind:
            self.overfull_ind = OverfullIndicator(ord=(self.i, 0))

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
        self.fig = plt.figure(figsize=(7.2, 9.6), layout="tight")
        # self.ax0 = self.fig.add_axes((0,0,1,1))
        # self.ax0.axis("off")
        # self.ax0.plot([0,1], [0.65, 0.65])

        """
        controls
        accuracy 3d
        img graph cbar
        """

        gs = GridSpec(nrows=3, ncols=1, figure=self.fig)
        gs.set_height_ratios((2, 38, 60))

        gs_controls = gs[0, 0].subgridspec(nrows=1, ncols=2)
        # ax_btn_cmap = self.fig.add_subplot(gs_controls[0, -3])
        # ax_box_dxdy = self.fig.add_subplot(gs_controls[0, -2])
        ax_btn_start = self.fig.add_subplot(gs_controls[0, -1])
        gs_controls.set_width_ratios((85, 15))

        gs_plots = gs[1, 0].subgridspec(nrows=2, ncols=2)
        gs_plots.set_height_ratios((20, 80))
        gs_plots.set_width_ratios((35, 65))
        ax_accuracy = self.fig.add_subplot(gs_plots[1, 0])
        ax_loss_projection = self.fig.add_subplot(gs_plots[:, 1], projection="3d")

        gs_graph = gs[2, 0].subgridspec(nrows=3, ncols=4)
        gs_graph.set_width_ratios((15, 80, 3, 2))
        gs_graph.set_height_ratios((30, 40, 30))
        ax_img = self.fig.add_subplot(gs_graph[:, 0])
        ax_graph = self.fig.add_subplot(gs_graph[:, 1])
        ax_cbar = self.fig.add_subplot(gs_graph[1, 2])

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
        for x_ord, layer in enumerate(self.layers):
            layer.previous = previous
            nodes1 = []
            nodes2 = []
            N = layer.width if not layer.overfull else self.maxwidth
            n = int(N / 2)

            if layer.overfull:
                for k in range(n):
                    y_ord = n - k
                    nodes1.append(Node((x_ord, y_ord)))
                    nodes2.append(Node((x_ord, -y_ord)))

                w = int(layer.width / 2)
                layer.set_visible_nodes(n=w - n, m=w + n - 1)

            else:
                centered = N % 2 == 1
                y_off = 0 if centered else -0.5
                for k in range(n):
                    y_ord = n - k + y_off
                    nodes1.append(Node((x_ord, y_ord)))
                    nodes2.append(Node((x_ord, -y_ord)))
                if centered:
                    nodes1.append(Node((x_ord, 0)))
                layer.set_visible_nodes(n=0, m=0)

            nodes2.reverse()
            layer.nodes = [*nodes1, *nodes2]

            previous = layer

        self.weights = np.empty(sum([layer.W_size for layer in self.layers]))

        # create node collection
        nodes = [node.node for layer in self.layers for node in layer.nodes]
        nodes.extend(
            [layer.overfull_ind.node for layer in self.layers if layer.overfull]
        )
        self.nodes = PatchCollection(
            nodes,
            transform=self.graph.transData,
            edgecolor="#bdbdbd",
            facecolor="white",
            linewidth=Node.radius,
            zorder=1,
        )

        # create overfull dot collection
        overfull_dots = [
            circle
            for layer in self.layers
            if layer.overfull
            for circle in layer.overfull_ind.circles
        ]
        self.overfull_dots = PatchCollection(
            overfull_dots, zorder=2, facecolor="#bdbdbd", edgecolor="none"
        )

        # create edge collection
        edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            _edges = []
            for node_in in layer2.nodes:
                for node_out in layer1.nodes:
                    _edges.append(Edge(node_out, node_in))
            edges.extend(_edges)
        self.edges = LineCollection(
            [edge.geometry for edge in edges], color="#bdbdbd", zorder=0, linewidth=2
        )

        # add colections to graph
        self.graph.add_collection(self.nodes)
        self.graph.add_collection(self.edges)
        self.graph.add_collection(self.overfull_dots)
        for layer in self.layers:
            self.graph.add_collection(layer.node_activations)

        bbox = max(self.layers, key=lambda l: l.width).bbox
        x0 = 0
        x1 = len(self.layers) * dx
        y0 = bbox.get_y()
        y1 = y0 + bbox.get_height()
        dataLim = ax_graph.dataLim
        dataLim.x0 = x0
        dataLim.x1 = x1
        dataLim.y0 = y0
        dataLim.y1 = y1

        # make controls
        # self.box_dxdy = TextBox(ax=ax_box_dxdy, label="(dx, dy)")
        # self.box_dxdy.label.set_fontsize("x-small")
        # self.box_dxdy.on_submit(self.update_geometry)
        # self.box_dxdy.set_val(f"{Context.dx}, {Context.dy}")
        self.btn_start = Button(ax=ax_btn_start, label="Start/Pause")
        self.btn_start.label.set_fontsize("x-small")

        self.ax_img = ax_img
        self.ax_img.axis("off")

        self.ax_loss_projection = ax_loss_projection

        self.ax_accuracy = ax_accuracy

        def func(values):
            val = max(abs(values.min()), abs(values.max()))
            return (-val, val)

        # setup cmap and cbar
        self.cmaps = cycle(
            (
                cmap(mpl.colormaps["tab20c"], norm=SymLogNorm(linthresh=0.03)),
                cmap(
                    cmap_red_green, norm=SymLogNorm(linthresh=0.03), compute_limits=func
                ),
            )
        )
        self.cmap = next(self.cmaps)

        self.cbar_ax = ax_cbar
        self.make_cbar()

        # add button for switching cbar
        # self.btn_cmap = Button(ax=ax_btn_cmap, label="Switch cmap")
        # self.btn_cmap.on_clicked(self.switch_cmap)

        # init weight colors
        self.update_weights()

        # self.fig.get_layout_engine().set(h_pad = 8/72)

    def make_cbar(self):
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=self.cmap.cmap),
            cax=self.cbar_ax,
            location="right",
            orientation="vertical",
            label=r"$\text{LogNorm}(W_{ij})$",
        )

    def switch_cmap(self, _=None):
        self.cmap = next(self.cmaps)
        self.make_cbar()
        self.update_weights()

    def update_weights(self):
        np.concat(
            tuple((layer.visible_edge_weights for layer in self.layers[1:])),
            axis=0,
            out=self.weights,
        )

        self.edges.set_colors(self.cmap.map(self.weights))

    def animate_ff(self):
        yield from AnimationIterator(
            flatten(
                map(lambda layer: layer.animate_activation(frames=30), self.layers),
                dim=2,
            )
        )

    def clear_activations(self):
        for layer in self.layers:
            layer.reset_activation_rad()
