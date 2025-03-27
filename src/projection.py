import numpy as np
from mlp1 import LayerWrapper, pipeline
from scipy.interpolate import RegularGridInterpolator
from collections import deque

def expand_range(x, xmin, xmax, padding=0):
    if x < xmin:
        return (x - padding, xmax)
    elif x > xmax:
        return (xmin, x + padding)
    return (xmin, xmax)


class ProjectionAxis:
    def __init__(self, arr, pos, num):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.arr_getter = lambda: None
        self.x0 = None
        self.xmin = None
        self.xmax = None
        self.update_x0()

    def update_x0(self):
        self.x0_prev = self.x0
        self.x0 = self.arr[*self.pos]

    def in_range(self, x):
        return self.xmin <= x and x <= self.xmax

    @property
    def lim(self):
        return (self.xmin, self.xmax)

    @lim.setter
    def lim(self, limits):
        (self.xmin, self.xmax) = limits

    def redraw(self):
        self.arr_ax = np.repeat(self.arr_getter()[np.newaxis], axis=0, repeats=self.num)
        self.axis = np.linspace(self.xmin, self.xmax, num=self.num)
        arr = np.moveaxis(self.arr_ax, source=0, destination=-1)
        arr[..., *self.pos, :] = self.axis


class ProjectionLayer(LayerWrapper):
    def __init__(self, layer):
        super().__init__(layer)
        self.axes = []
        self.W_ = np.copy(self.layer.W)
        self.W_getter = lambda: self.W_
        self.W_redraw = [lambda: np.copyto(src=self.layer.W, dst=self.W_)]

        self.b_ = np.copy(self.layer.b)
        self.b_getter = lambda: self.b_
        self.b_redraw = [lambda: np.copyto(src=self.layer.b, dst=self.b_)]

    def add_axes(self, W=None, b=None):
        if W:
            self.W_redraw = []
            getter = lambda: self.layer.W
            for axis in reversed(W):
                self.axes.append(axis)
                axis.arr_getter = getter
                self.W_redraw.append(lambda ax=axis: ax.redraw())
                getter = lambda ax=axis: ax.arr_ax
            self.W_getter = lambda ax=axis: ax.arr_ax

        if b:
            self.b_redraw = []
            getter = lambda: self.layer.b
            for axis in reversed(b):
                self.axes.append(axis)
                axis.arr_getter = getter
                self.b_redraw.append(lambda ax=axis: ax.redraw())
                getter = lambda ax=axis: ax.arr_ax
            self.b_getter = lambda ax=axis: ax.arr_ax

            W_getter = self.W_getter
            self.W_getter = lambda: np.expand_dims(
                W_getter(), axis=(1 if W else tuple(range(len(b))))
            )

    def redraw(self):
        for fn in [*self.W_redraw, *self.b_redraw]:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @W.setter
    def W(self, _):
        return

    @property
    def b(self):
        return self.b_getter()

    @b.setter
    def b(self, _):
        return


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y

        self.axes = [ax for layer in layers for ax in layer.axes]

        self.layers = list(layers)
        self.pipeline = pipeline(layers)

    def compute(self):
        self.pipeline.feedforward(self.X)
        self.pipeline.run_model(Y=self.Y)

    def redraw(self):
        """
        Refresh W, b arrays for all layers to mirror current W and b having undergone optimization

        For projection layers with axes, compute W and b with new values and axis ranges
        """
        for proj_layer in self.layers:
            proj_layer.redraw()
        self.compute()

    @property
    def grid(self):
        return np.squeeze(self.pipeline.model.loss)


class ProjectionView:
    def __init__(self, ax, proj_layers, X, Y, update_interval=10):
        self.update_interval = update_interval
        self.i = 0
        self.max_values = deque(maxlen=5)
        self.min_values = deque(maxlen=5)

        self.ax = ax
        self.ax.set_zlim(0)
        self.ax.set_xlabel("ax 1")
        self.ax.set_ylabel("ax 2")
        self.ax.set_zlabel("loss")

        self.grid = ProjectionGrid(layers=proj_layers, X=X, Y=Y)
        [self.ax1, self.ax2] = self.grid.axes
        self.domain = ProjectionDomain(self.ax1, self.ax2)

        self.surface = None
        self.scatter = None

        self.redraw()

    def draw_frame(self):
        self.i += 1
        redraw = self.i >= self.update_interval

        self.ax1.update_x0()
        self.ax2.update_x0()
        x1, x2 = (self.ax1.x0, self.ax2.x0)

        if not self.domain.in_domain(x1, x2):
            redraw = True
            self.domain.update(x1, x2)

        if redraw:
            self.redraw()

        self.draw_point(x1, x2)

    def interpolate(self, x1, x2):
        [x3] = self.interp([x1, x2])
        return (x1, x2, x3)

    def draw_point(self, x1, x2):
        if self.scatter:
            self.scatter.remove()
        self.scatter = self.ax.scatter(*self.interpolate(x1, x2), color="red")

    def redraw(self):
        self.i = 0
        ax1, ax2 = (self.ax1, self.ax2)
        self.grid.redraw()
        grid = self.grid.grid
        self.interp = RegularGridInterpolator(points=(ax1.axis, ax2.axis), values=grid)
        if self.surface:
            self.surface.remove()
        X, Y = np.meshgrid(ax1.axis, ax2.axis, copy=False, indexing="ij")
        self.surface = self.ax.plot_surface(
            X, Y, Z=grid, alpha=0.3, edgecolor="royalblue", color="gray"
        )
        
        self.ax.set_xlim(*ax1.lim)
        self.ax.set_ylim(*ax2.lim)
        self.min_values.append(np.min(grid))
        self.max_values.append(np.max(grid))
        self.ax.set_zlim(max(0, min(self.min_values)-0.5), max(self.max_values))

class ProjectionDomain:
    def __init__(self, ax1, ax2, d1=None, d2=None):
        self.ax1 = ax1
        self.ax2 = ax2

        x1 = self.ax1.x0
        d1 = d1 or abs(x1) or 1
        self.ax1.lim = (x1 - d1, x1 + d1)

        x2 = self.ax2.x0
        d2 = d2 or abs(x2) or 1
        self.ax2.lim = (x2 - d2, x2 + d2)

    def in_domain(self, x1, x2):
        return self.ax1.in_range(x1) and self.ax2.in_range(x2)

    def update(self, x1, x2):
        for ax, x in ((self.ax1, x1), (self.ax2, x2)):
            if not ax.in_range(x):
                pad = abs(ax.x0 - ax.x0_prev)
                ax.lim = expand_range(x, *ax.lim, pad)
