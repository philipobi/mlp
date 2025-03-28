import numpy as np
from main import init_training_data
import os.path
from time import sleep
from projection import ProjectionAxis, ProjectionLayer, ProjectionView
from mlp1 import Layer, Training, pipeline, LayerWrapper
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main():
    dims = [28 * 28, 16, 16, 10]
    path = "params/small_50epochs_93percent"
    layers = [Layer(i, j) for i, j in zip(dims[:-1], dims[1:])]
    if 0:
        for i, layer in enumerate(layers, start=1):
            layer.load_params(
                wpath=os.path.join(path, f"W{i}.npy"),
                bpath=os.path.join(path, f"b{i}.npy"),
            )

    (N, it, valset) = init_training_data(
        "data/train.csv", batchsize=20, valsetsize=20, epochs=10
    )

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    
    X, Y = valset

    proj_layers = [ProjectionLayer(layer) for layer in layers]

    proj_layer = proj_layers[-1]
    proj_layer.add_axes(
        b=(
            ProjectionAxis(arr=proj_layer.layer.b, pos=(0,), num=100),
        ),
        W=(
            ProjectionAxis(arr=proj_layer.layer.W, pos=(7,0), num=100),
        ),
    )

    view = ProjectionView(ax, proj_layers, X=X, Y=Y, update_interval=50)

    training = Training(layers, it, valset, alpha=0.03)

    def update(_):
        training.train_minibatch()
        view.draw_frame()
        # input()

    ani = FuncAnimation(fig, func=update, cache_frame_data=False, interval=0)

    plt.show()

    return
    while not training.completed:
        training.train_minibatch()
        sleep(0.1)

    # print("djdw")
    # for djdw in training.training_pipeline.dJdW:
    #     print(djdw)
    # print("output")
    # print(training.training_pipeline.model.A)


main()
