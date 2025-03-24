import numpy as np
from mlp1 import LayerWrapper, mlogit, ff, relu

class ProjectionAxis:
    def __init__(self, arr, pos, num, d):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.d = d
        self.redraw_ax(all=True)

    @property
    def in_range(self):
        x0 = self.arr[..., *self.pos]
        return x0 >= self.xmin and x0 <= self.xmax

    def redraw_ax(self, all=False):
        if all:
            self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        x0 = self.arr[..., *self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[:, ..., *self.pos] = np.linspace(self.xmin, self.xmax, num=self.num)

class ProjectionLayer(LayerWrapper):
    def __init__(self, layer, W=None, b=None):
        super().__init__(layer)

        self.redraw_hooks = []
        self.ax_W = W
        self.ax_b = b

        if W is None:
            self.W_ = np.copy(self.layer.W)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.W, dst=self.W_))
            self.W_getter = lambda: self.W_
        else:
            self.redraw_hooks.append(lambda: W.redraw_ax(all=True))
            self.W_getter = lambda: W.arr_ax

        if b is None:
            self.b_ = np.copy(self.layer.b)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.b, dst=self.b_))
            self.b_getter = lambda: self.b_
        else:
            self.redraw_hooks.append(lambda: b.redraw_ax(all=True))
            self.b_getter = lambda: b.arr_ax

    def redraw_all(self):
        for fn in self.redraw_hooks:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @property
    def b(self):
        return self.b_getter()


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y
        self.model = mlogit()

        self.axes = []
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.layers = layers[:-1]
        self.layer_out = layers[-1]

        for layer in self.layers:
            layer.ff = ff()
            layer.act = relu()

        self.layer_out.ff = ff()

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        A = self.X
        for layer in self.layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(ff.Z)
            A = layer.act.A

        layer = self.layer_out
        layer.ff.run(A, layer.W, layer.b)
        self.model.run(layer.Z, A, self.Y, bprop=False)

    @property
    def grid(self):
      class ProjectionAxis:
    def __init__(self, arr, pos, num, d):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.d = d
        self.redraw_ax(all=True)

    @property
    def in_range(self):
        x0 = self.arr[..., *self.pos]
        return x0 >= self.xmin and x0 <= self.xmax

    def redraw_ax(self, all=False):
        if all:
            self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        x0 = self.arr[..., *self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[:, ..., *self.pos] = np.linspace(self.xmin, self.xmax, num=self.num)





class ProjectionLayer(LayerWrapper):
    def __init__(self, layer, W=None, b=None):
        super().__init__(layer)

        self.redraw_hooks = []
        self.ax_W = W
        self.ax_b = b

        if W is None:
            self.W_ = np.copy(self.layer.W)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.W, dst=self.W_))
            self.W_getter = lambda: self.W_
        else:
            self.redraw_hooks.append(lambda: W.redraw_ax(all=True))
            self.W_getter = lambda: W.arr_ax

        if b is None:
            self.b_ = np.copy(self.layer.b)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.b, dst=self.b_))
            self.b_getter = lambda: self.b_
        else:
            self.redraw_hooks.append(lambda: b.redraw_ax(all=True))
            self.b_getter = lambda: b.arr_ax

    def redraw_all(self):
        for fn in self.redraw_hooks:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @property
    def b(self):
        return self.b_getter()


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y
        self.model = mlogit()

        self.axes = []
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.layers = layers[:-1]
        self.layer_out = layers[-1]

        for layer in self.layers:
            layer.ff = ff()
            layer.act = relu()

        self.layer_out.ff = ff()

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        A = self.X
        for layer in self.layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(ff.Z)
            A = layer.act.A

        layer = self.layer_out
        layer.ff.run(A, layer.W, layer.b)
        self.model.run(layer.Z, A, self.Y, bprop=False)

    @property
    def grid(self):
      class ProjectionAxis:
    def __init__(self, arr, pos, num, d):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.d = d
        self.redraw_ax(all=True)

    @property
    def in_range(self):
        x0 = self.arr[..., *self.pos]
        return x0 >= self.xmin and x0 <= self.xmax

    def redraw_ax(self, all=False):
        if all:
            self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        x0 = self.arr[..., *self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[:, ..., *self.pos] = np.linspace(self.xmin, self.xmax, num=self.num)





class ProjectionLayer(LayerWrapper):
    def __init__(self, layer, W=None, b=None):
        super().__init__(layer)

        self.redraw_hooks = []
        self.ax_W = W
        self.ax_b = b

        if W is None:
            self.W_ = np.copy(self.layer.W)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.W, dst=self.W_))
            self.W_getter = lambda: self.W_
        else:
            self.redraw_hooks.append(lambda: W.redraw_ax(all=True))
            self.W_getter = lambda: W.arr_ax

        if b is None:
            self.b_ = np.copy(self.layer.b)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.b, dst=self.b_))
            self.b_getter = lambda: self.b_
        else:
            self.redraw_hooks.append(lambda: b.redraw_ax(all=True))
            self.b_getter = lambda: b.arr_ax

    def redraw_all(self):
        for fn in self.redraw_hooks:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @property
    def b(self):
        return self.b_getter()


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y
        self.model = mlogit()

        self.axes = []
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.layers = layers[:-1]
        self.layer_out = layers[-1]

        for layer in self.layers:
            layer.ff = ff()
            layer.act = relu()

        self.layer_out.ff = ff()

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        A = self.X
        for layer in self.layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(ff.Z)
            A = layer.act.A

        layer = self.layer_out
        layer.ff.run(A, layer.W, layer.b)
        self.model.run(layer.Z, A, self.Y, bprop=False)

    @property
    def grid(self):
      class ProjectionAxis:
    def __init__(self, arr, pos, num, d):
        self.arr = arr
        self.pos = pos
        self.num = num
        self.d = d
        self.redraw_ax(all=True)

    @property
    def in_range(self):
        x0 = self.arr[..., *self.pos]
        return x0 >= self.xmin and x0 <= self.xmax

    def redraw_ax(self, all=False):
        if all:
            self.arr_ax = np.repeat(self.arr[np.newaxis], axis=0, repeats=self.num)
        x0 = self.arr[..., *self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[:, ..., *self.pos] = np.linspace(self.xmin, self.xmax, num=self.num)





class ProjectionLayer(LayerWrapper):
    def __init__(self, layer, W=None, b=None):
        super().__init__(layer)

        self.redraw_hooks = []
        self.ax_W = W
        self.ax_b = b

        if W is None:
            self.W_ = np.copy(self.layer.W)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.W, dst=self.W_))
            self.W_getter = lambda: self.W_
        else:
            self.redraw_hooks.append(lambda: W.redraw_ax(all=True))
            self.W_getter = lambda: W.arr_ax

        if b is None:
            self.b_ = np.copy(self.layer.b)
            self.redraw_hooks.append(lambda: np.copyto(src=self.layer.b, dst=self.b_))
            self.b_getter = lambda: self.b_
        else:
            self.redraw_hooks.append(lambda: b.redraw_ax(all=True))
            self.b_getter = lambda: b.arr_ax

    def redraw_all(self):
        for fn in self.redraw_hooks:
            fn()

    @property
    def W(self):
        return self.W_getter()

    @property
    def b(self):
        return self.b_getter()


class ProjectionGrid:
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y
        self.model = mlogit()

        self.axes = []
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.layers = layers[:-1]
        self.layer_out = layers[-1]

        for layer in self.layers:
            layer.ff = ff()
            layer.act = relu()

        self.layer_out.ff = ff()

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        A = self.X
        for layer in self.layers:
            layer.ff.run(A, layer.W, layer.b)
            layer.act.run(ff.Z)
            A = layer.act.A

        layer = self.layer_out
        layer.ff.run(A, layer.W, layer.b)
        self.model.run(layer.Z, A, self.Y, bprop=False)

    @property
    def grid(self):
      return self.model.loss