import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    SymLogNorm,
    ListedColormap,
)
from matplotlib.widgets import Button
import matplotlib as mpl
import numpy as np
from itertools import cycle, zip_longest
from utils import Iterator, flatten, cmap_red_green, Array
from matplotlib.lines import Line2D

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
        for i, plot in enumerate(self.plots, start=1):
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
        (self.x_ord, self.y_ord) = ord
        self.xy = (self.x_ord * self.ctx.dx, self.y_ord * self.ctx.dy)

        self.node = Circle(xy=self.xy, radius=Node.radius, **kwargs)
        self.activation = Circle(xy=self.xy, radius=0, **kwargs)


class OverfullIndicator:
    ctx = Context

    def __init__(self, ord):
        x_ord, y_ord = ord
        self.xy = (x_ord * self.ctx.dx, y_ord * self.ctx.dy)

        x, y = self.xy
        r = Node.radius
        self.node = Circle(xy=self.xy, radius=r)
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

        self.overfull = False
        self.overfull_ind = []

        self.nodes = []

        self.node_activations = PatchCollection(
            [],
            transform=self.ctx.graph.transData,
            facecolor="#bdbdbd",
            linewidth=0,
            zorder=2,
        )

        self._visible_edge_weights = lambda W: None
        self._visible_activations = lambda A: None

    def init_components(self, nodes):
        """
        Make auxiliary objects for layer, such as overfull dots, activations and bbox
        """

        self.nodes = nodes
        self.node_activations.objects = [node.activation for node in self.nodes]

        r = Node.radius
        pad = r / 2

        if not self.overfull:
            x0, y0 = self.nodes[0].xy
            x1, y1 = self.nodes[-1].xy
        else:
            self.overfull_ind.append(
                ind0 := OverfullIndicator(ord=(self.i, self.nodes[0].y_ord - 1))
            )
            self.overfull_ind.append(
                ind1 := OverfullIndicator(ord=(self.i, self.nodes[-1].y_ord + 1))
            )
            x0, y0 = ind0.xy
            x1, y1 = ind1.xy

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
                linewidth=2.5,
            )
        )

    def set_activation_rad(self, radii):
        for circle, radius in zip(self.node_activations.objects, radii):
            circle.set_radius(radius)

    def reset_activation_rad(self):
        for circle in self.node_activations.objects:
            circle.set_radius(0)
        self.node_activations.update_objects()

    def animate_activation(self):
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
                frames=50,
            ),
            AnimationSpec(func=func, transition=transitionLinear, frames=30),
            AnimationSpec(
                func=lambda i: self.bbox.set_alpha(1 - i),
                transition=transitionLinear,
                frames=50,
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

        # mpl.rcParams["toolbar"] = "None"
        mpl.rcParams["axes3d.mouserotationstyle"] = "azel"
        # create mpl objects
        fig = self.fig = plt.figure(figsize=(7.2, 9.6))

        """
        controls
        accuracy 3d
        img graph cbar
        """

        ax_btn_start = fig.add_axes((0.9, 0.97, 0.05, 0.02))
        ax_btn_reset = fig.add_axes((0.85, 0.97, 0.05, 0.02))

        ax_accuracy = fig.add_axes((0.08, 0.64, 0.25, 0.2))
        ax_loss_projection = fig.add_axes((0.3, 0.61, 0.75, 0.35), projection="3d")

        ax_img = fig.add_axes((0.05, 0.01, 0.15, 0.59))
        ax_graph = fig.add_axes((0.12, 0.00, 0.8, 0.59))
        ax_cbar = fig.add_axes((0.93, 0.2, 0.02, 0.2))

        x = 0.37
        y = 0.58
        self.div_lines = [
            fig.add_artist(Line2D([0, 1], [y, y], color="black")),
            fig.add_artist(Line2D([x, x], [y, 1], color="black")),
        ]

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

            N = layer.width if not layer.overfull else self.maxwidth
            n = int(N / 2)
            centered = N % 2 == 1
            y_off = 0 if centered else 0.5

            nodes = [
                Node((x_ord, y_ord + y_off))
                for y_ord in range(-n, n + (1 if centered else 0))
            ]

            if layer.overfull:
                w = int(layer.width / 2)
                i_n = w - n
                i_m = i_n + N
                layer.set_visible_nodes(n=i_n, m=i_m)
            else:
                layer.set_visible_nodes(n=0, m=0)

            if x_ord == 0:
                i_n = 15 * 28 + 15 - n
                i_m = i_n + N
                layer.set_visible_nodes(n=i_n, m=i_m)

            layer.init_components(nodes)

            previous = layer

        self.weights = np.empty(sum([layer.W_size for layer in self.layers]))

        # create node collection
        nodes = [node.node for layer in self.layers for node in layer.nodes]
        nodes.extend(
            [
                overfull_ind.node
                for layer in self.layers
                for overfull_ind in layer.overfull_ind
            ]
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
            for overfull_ind in layer.overfull_ind
            for circle in overfull_ind.circles
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

        #make controls
        self.btn_start = Button(ax=ax_btn_start, label="Run")
        self.btn_start.label.set_fontsize("x-small")

        self.btn_reset = Button(ax=ax_btn_reset, label="Res")
        self.btn_reset.label.set_fontsize("x-small")
        
        #set various plot parameters
        bbox = max(self.layers, key=lambda l: l.width).bbox
        y0 = bbox.get_y()
        y1 = y0 + bbox.get_height()
        x0 = self.layers[0].bbox.get_x()
        bbox = self.layers[-1].bbox
        x1 = bbox.get_x() + bbox.get_width()

        self.graph.set_xlim(x0, x1 + 2.5)
        self.graph.set_ylim(y0 - 2, y1 + 2)
        
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

        # init weight colors
        self.update_weights()


    def make_cbar(self):
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=self.cmap.cmap),
            cax=self.cbar_ax,
            location="left",
            orientation="vertical",
            label="Weight values",
        )
        self.cbar.set_ticks(ticks=[0, .5, 1], labels=['-', '0', '+'])

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
                map(lambda layer: layer.animate_activation(), self.layers),
                dim=2,
            )
        )

    def clear_activations(self):
        for layer in self.layers:
            layer.reset_activation_rad()

    def set_div_alpha(self, val):
        for line in self.div_lines:
            line.set_alpha(val)
