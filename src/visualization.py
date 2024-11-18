import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.widgets import Slider


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
    def __init__(self, node1, node2, incoming_layer):
        self.node1 = node1
        self.node2 = node2
        self.incoming_layer = incoming_layer

    @property
    def geometry(self):
        return [self.node1.xy, self.node2.xy]


class Layer:
    def __init__(self, layer):
        self.width = layer.width
        # self.A = layer.A
        self.overfull = False
        self.nodes = []

    def add_nodes(self, nodes):
        for node in nodes:
            node.parent = self
        self.nodes.extend(nodes)


class LayerInterface:
    def __init__(self, w):
        self.width = w


class MLPInterface:
    layers = [
        LayerInterface(28 * 28),
        LayerInterface(16),
        LayerInterface(16),
        LayerInterface(10),
    ]


class MLPVisualization:
    maxwidth = 20

    def __init__(self, mlp, dx, dy, r):
        self.fig, ax = plt.subplots(3, 1, height_ratios=[8, 1, 1])
        (self.ax1, self.ax2, self.ax3) = ax

        self.layers = [Layer(layer) for layer in mlp.layers]
        n = len(self.layers)

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
                for k in range(c if centered else c - 1):
                    y_ord = c - k + y_offset
                    nodes1.append(Node(i=k, y_ord=y_ord))
                    nodes2.append(Node(i=(layer.width - 1) - k, y_ord=-y_ord))
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
            for node1 in layer1.nodes:
                for node2 in layer2.nodes:
                    self.edges.append(Edge(node1, node2, incoming_layer=layer2))

        self.patch_c = PatchCollection([], transform=self.ax1.transData)
        self.line_c = LineCollection([], color="gray", zorder=-1)

        self.ax1.add_collection(self.patch_c)
        self.ax1.add_collection(self.line_c)

        self.dx_slider = Slider(
            ax=self.ax2, label="dx", valmin=1, valmax=10, valinit=self.dx
        )
        self.dy_slider = Slider(
            ax=self.ax3, label="dy", valmin=1, valmax=10, valinit=self.dy
        )

        self.dx_slider.on_changed(self.update)
        self.dy_slider.on_changed(self.update)

        self.draw_nodes()
        self.ax1.axis("equal")

        self.fig.tight_layout()
        plt.show()

    def update(self, val):
        self.dx = self.dx_slider.val
        self.dy = self.dy_slider.val
        self.draw_nodes()
        x_min = self.x_minnode.geometry.get_extents().xmin
        x_max = self.x_maxnode.geometry.get_extents().xmax
        y_max = self.y_maxnode.geometry.get_extents().ymax
        y_min = -y_max
        min_ = min(x_min, y_min)
        max_ = max(x_max, y_max)
        self.ax1.set_xlim(min_, max_)
        self.ax1.set_ylim(min_, max_)
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
