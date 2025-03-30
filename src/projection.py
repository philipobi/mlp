import numpy as np
from mlp import LayerWrapper, pipeline
from scipy.interpolate import RegularGridInterpolator
from collections import deque


def linspace(start, stop, num, out):
    d = (stop - start) / num
    for i in range(num):
        out[i] = start + i * d


class LimitsControl:
    def __init__(self, n_samples=10, default_lim=(0, 1000)):
        self.xmin_def, self.xmax_def = default_lim
        self.xmin_vals = deque(maxlen=n_samples)
        self.xmax_vals = deque(maxlen=n_samples)

    def update(self, xmin, xmax):
        xmin = max(xmin, self.xmin_def)
        xmax = min(xmax, self.xmax_def)
        self.xmin_vals.append(xmin)
        self.xmax_vals.append(xmax)
        return (min(self.xmin_vals), max(self.xmax_vals))


class DescentPath:
    def __init__(self, ax, n_samples=200):
        self.n = n_samples
        self.arr0 = np.empty((2 * self.n, 3))
        self.arr1 = np.empty_like(self.arr0)
        self.i = 0
        self.path = ax.plot(xs=[], ys=[], zs=[], color="red", lw=2)[0]

    def reduce(self):
        self.arr1[: self.n] = self.arr0[-self.n :]
        (self.arr0, self.arr1) = (self.arr1, self.arr0)
        self.i = self.n

    def redraw(self, interp):
        self.arr[2][:] = interp(self.arr0[: self.i, :2])

    def append(self, x, y, z):
        if self.i < 2 * self.n:
            self.arr0[self.i] = [x, y, z]
            self.i += 1
        else:
            self.reduce()
            self.append(x, y, z)

    def update(self, x, y, z):
        self.append(x, y, z)
        self.path.set_data_3d(self.arr)

    @property
    def arr(self):
        return self.arr0[: self.i].T


class ProjectionAxis:
    def __init__(self, arr, pos, num):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.arr_getter = lambda: None
        self.x0 = None
        self.xmin = None
        self.xmax = None
        self.axis = np.empty(self.num)
        self.update_x0()

        def func(arr):
            np.copyto(src=arr, dst=self.arr_ax)

        def func_(arr):
            self.arr_ax = np.copy(arr)
            self.copy_arr_ax = func

        self.copy_arr_ax = func_

    def update_x0(self):
        self.x0_prev = self.x0
        self.x0 = self.arr[*self.pos]

    @property
    def lim(self):
        return (self.xmin, self.xmax)

    @lim.setter
    def lim(self, limits):
        (self.xmin, self.xmax) = limits

    def redraw(self):
        arr = self.arr_getter()
        arr = np.broadcast_to(arr, shape=(self.num, *arr.shape))
        self.copy_arr_ax(arr)

        linspace(self.xmin, self.xmax, num=self.num, out=self.axis)
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

        def func():
            np.square(self.grid, out=self.grid_scaled)

        def func_():
            self.grid_scaled = np.square(self.grid)
            self.compute_log = func

        self.compute_scaled = func_

    def compute(self):
        self.pipeline.feedforward(self.X)
        self.pipeline.run_model(Y=self.Y)
        # self.compute_scaled()

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


class ProjectionDomain:
    def __init__(self, ax1, ax2, d=None):
        self.ax1 = ax1
        self.ax2 = ax2

        for ax in (self.ax1, self.ax2):
            x = ax.x0
            d = d or abs(x) / 8 or 1
            ax.lim = (x - d, x + d)

    def update(self, x1, x2):
        redraw = False
        for ax, x in ((self.ax1, x1), (self.ax2, x2)):
            xmin, xmax = ax.lim
            d = xmax - xmin
            margin = d / 16
            pad = d / 8
            if xmax - x < margin:
                xmax = x + pad
                redraw = True
            if x - xmin < margin:
                xmin = x - pad
                redraw = True
            ax.lim = (xmin, xmax)
        return redraw


class ProjectionView:
    def __init__(self, ax, proj_layers, X, Y, update_interval=10):
        self.update_interval = update_interval
        self.i = 0

        self.ax = ax
        self.ax.set_zlim(0)
        self.ax.set_xlabel("ax 1")
        self.ax.set_ylabel("ax 2")
        self.ax.set_zlabel("loss")

        self.grid = ProjectionGrid(layers=proj_layers, X=X, Y=Y)
        [self.ax1, self.ax2] = self.grid.axes
        self.domain = ProjectionDomain(self.ax1, self.ax2)
        self.zlim_domain = LimitsControl()

        self.descent_path = DescentPath(self.ax, n_samples=500)
        self.surface = None

        self.redraw()

    def draw_frame(self):
        self.i += 1

        self.ax1.update_x0()
        self.ax2.update_x0()
        x, y = (self.ax1.x0, self.ax2.x0)

        if self.domain.update(x, y) or self.i >= self.update_interval:
            self.redraw()

        self.descent_path.update(*self.interpolate(x, y))

    def interpolate(self, x1, x2):
        [x3] = self.interp([x1, x2])
        return (x1, x2, x3)

    def redraw(self):
        self.i = 0
        ax1, ax2 = (self.ax1, self.ax2)
        self.grid.redraw()
        grid = self.grid.grid
        self.interp = RegularGridInterpolator(points=(ax1.axis, ax2.axis), values=grid)
        self.descent_path.redraw(self.interp)
        if self.surface:
            self.surface.remove()
        X, Y = np.meshgrid(ax1.axis, ax2.axis, copy=False, indexing="ij")
        self.surface = self.ax.plot_surface(
            X, Y, Z=grid, cmap="coolwarm", alpha=0.5, axlim_clip=True
        )

        self.ax.set_xlim(*ax1.lim)
        self.ax.set_ylim(*ax2.lim)

        self.ax.set_zlim(max(np.min(grid), 0), min(np.max(grid), 1000))
