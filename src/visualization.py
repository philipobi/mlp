import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection, LineCollection


class Node:
    def __init__(self, xy, i, **kwargs):
        self.i = i
        self.xy = xy
        self.geometry = Circle(xy=xy, **kwargs)


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.geometry = [node1.xy, node2.xy]


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

    def __init__(self, mlp, x_width, y_width, x_pad, y_pad):
        self.layers = [Layer(layer) for layer in mlp.layers]
        n = len(self.layers)

        x_width -= 2 * x_pad
        y_width -= 2 * y_pad

        m = 0
        for layer in self.layers:
            if layer.width > self.maxwidth:
                layer.overfull = True
                m = self.maxwidth
            elif layer.width > m:
                m = layer.width

        # y_spacing = r
        r = y_width / (3 * m - 1)
        y_mid = y_width / 2

        x_offset = r
        x_spacing = (x_width - 2 * r) / (n - 1)

        for j, layer in enumerate(self.layers):
            x = x_offset + j * x_spacing
            nodes1 = []
            nodes2 = []
            N = layer.width if not layer.overfull else self.maxwidth
            centered = N % 2 == 1
            y_offset = 0 if centered else -3 * r / 2
            c = int(N / 2)

            if layer.overfull:
                for i in range(c if centered else c - 1):
                    y = (c - i) * 3 * r + y_offset
                    nodes1.append(Node(xy=(x, y_mid + y), radius=r, i=i))
                    nodes2.append(
                        Node(xy=(x, y_mid - y), radius=r, i=layer.width - 1 - i)
                    )
            else:
                for i in range(c):
                    y = (c - i) * 3 * r + y_offset
                    nodes1.append(Node(xy=(x, y_mid + y), radius=r, i=i))
                    nodes2.append(
                        Node(xy=(x, y_mid - y), radius=r, i=layer.width - 1 - i)
                    )
                if centered:
                    nodes1.append(Node(xy=(x, y_mid), radius=r, i=i + 1))

            nodes2.reverse()
            layer.add_nodes(nodes1)
            layer.add_nodes(nodes2)

        edges = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            for i, node1 in enumerate(layer1.nodes):
                for j, node2 in enumerate(layer2.nodes):

                    edges.append(Edge(node1, node2))

        self.edges = edges

        circles = PatchCollection(
            [node.geometry for layer in self.layers for node in layer.nodes]
        )

        lines = LineCollection(
            [edge.geometry for edge in self.edges], color="gray", zorder=-1
        )

        fig, ax = plt.subplots()
        ax.add_collection(circles)
        ax.add_collection(lines)
        ax.set_xlim(-5, 15)
        ax.set_ylim(-5, 15)
        ax.set_aspect("equal")
        plt.show()


def main():
    vis = MLPVisualization(mlp=MLPInterface, x_width=10, y_width=15, x_pad=1, y_pad=1)


main()
