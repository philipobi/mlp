import numpy as np
from mlp1 import LayerWrapper, pipeline
from scipy.interpolate import RegularGridInterpolator


class ProjectionAxis:
    def __init__(self, arr, pos, num, d=None):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.x0 = None
        self.d = d
        self.update_x0()
        self.redraw_ax()

    def update_x0(self):
        self.x0_prev = self.x0
        self.x0 = self.arr[*self.pos]

    @property
    def in_range(self):
        return self.x0 >= self.xmin and self.x0 <= self.xmax

    def redraw_ax(self):
        self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        d = (
            abs(self.x0)
            if self.x0_prev is None
            else abs(self.num * self.x0 - self.num * self.x0_prev)
        )
        d = self.d or d or 1
        self.xmin = self.x0 - d
        self.xmax = self.x0 + d
        self.axis = np.linspace(self.xmin, self.xmax, num=self.num)
        self.arr_ax[..., *self.pos].reshape(-1, copy=False)[:] = self.axis


class ProjectionLayer(LayerWrapper):
    def __init__(self, layer):
        super().__init__(layer)
        self.ax_W = None
        self.ax_b = None

        self.W_ = np.copy(self.layer.W)
        self.W_getter = lambda: self.W_
        self.W_redraw = lambda: np.copyto(src=self.layer.W, dst=self.W_)

        self.b_ = np.copy(self.layer.b)
        self.b_getter = lambda: self.b_
        self.b_redraw = lambda: np.copyto(src=self.layer.b, dst=self.b_)

    def add_axes(self, W=None, b=None):
        self.ax_W = W
        self.ax_b = b

        if W:
            self.W_redraw = lambda: W.redraw_ax()
            self.W_getter = lambda: W.arr_ax

        if b:
            self.b_redraw = lambda: b.redraw_ax()
            self.b_getter = lambda: b.arr_ax
            W_getter = self.W_getter
            self.W_getter = lambda: np.expand_dims(W_getter(), axis=(1 if W else 0))

    def redraw(self):
        self.W_redraw()
        self.b_redraw()

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
    def __init__(self, layers, X, Y, update_interval=10):
        self.update_interval = update_interval
        self.i = 0
        self.X = X
        self.Y = Y

        self.axes = []

        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.pipeline = pipeline(layers)
        self.layers = layers

        self.compute()

    def compute(self):
        self.pipeline.feedforward(self.X)
        self.pipeline.run_model(Y=self.Y)

    def redraw(self):
        self.i = 0
        for proj_layer in self.layers:
            proj_layer.redraw()
        self.compute()

    def iter(self):
        self.i += 1

        redraw = False

        for ax in self.axes:
            ax.update_x0()
            if not ax.in_range:
                redraw = True

        if self.i == self.update_interval:
            redraw = True

        return redraw

    @property
    def grid(self):
        return np.squeeze(self.pipeline.model.loss)


class ProjectionView:
    def __init__(self, ax, ax1, ax2, grid):
        self.ax = ax
        self.surface = None
        self.scatter = None
        self.redraw(ax1, ax2, grid)

    def interpolate(self, x, y):
        [z] = self.interp([x, y])
        return (x, y, z)

    def draw_point(self, x, y):
        if self.scatter:
            self.scatter.remove()
        self.scatter = self.ax.scatter(*self.interpolate(x, y), color="red")

    def redraw(self, ax1, ax2, grid):
        self.interp = RegularGridInterpolator(points=(ax1.axis, ax2.axis), values=grid)
        if self.surface:
            self.surface.remove()
        X, Y = np.meshgrid(ax1.axis, ax2.axis, copy=False, indexing="ij")
        self.surface = self.ax.plot_surface(
            X, Y, Z=grid, alpha=0.3, edgecolor="royalblue", color="gray"
        )
        self.ax.set_xlim(ax1.xmin, ax1.xmax)
        self.ax.set_ylim(ax2.xmin, ax2.xmax)
        zmin, zmax = (np.min(grid), np.max(grid))
        self.ax.set_zlim(zmin, zmax)
