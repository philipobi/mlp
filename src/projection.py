import numpy as np
from mlp1 import LayerWrapper, pipeline

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
        x0 = self.arr[*self.pos]
        self.xmin = x0 - self.d
        self.xmax = x0 + self.d
        self.arr_ax[..., *self.pos].reshape(-1, copy=False)[:] = np.linspace(self.xmin, self.xmax, num=self.num)

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
            self.W_redraw = lambda: W.redraw_ax(all=True)
            self.W_getter = lambda: W.arr_ax

        if b:
            self.b_redraw = lambda: b.redraw_ax(all=True)
            self.b_getter = lambda: b.arr_ax
            W_getter = self.W_getter
            self.W_getter = lambda: np.expand_dims(W_getter(), axis=(1 if W else 0))

    def redraw_all(self):
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
    def __init__(self, layers, X, Y):
        self.X = X
        self.Y = Y

        self.axes = []
        
        self.axes.extend([ax for layer in layers if (ax := layer.ax_W) is not None])
        self.axes.extend([ax for layer in layers if (ax := layer.ax_b) is not None])

        self.pipeline = pipeline(layers)
        self.layers = layers

    def compute(self):
        for ax in self.axes:
            if not ax.in_range:
                ax.redraw()

        self.pipeline.feedforward(self.X)
        self.pipeline.run_model(Y=self.Y)

    def redraw_all(self):
        for proj_layer in self.layers:
            proj_layer.redraw_all()

    @property
    def grid(self):
        return self.pipeline.model.loss