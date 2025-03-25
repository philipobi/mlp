import numpy as np
from main import init_training_data
import os.path
from time import sleep
from optimize import adam
from projection import ProjectionAxis, ProjectionLayer, ProjectionGrid
from mlp1 import Layer, Training

def main():
    dims = [28 * 28, 16, 16, 10]
    path = "params/small_50epochs_93percent"
    layers = [Layer(i, j) for i, j in zip(dims[:-1], dims[1:])]

    (N, it, valset) = init_training_data(
        "data/train.csv", batchsize=20, valsetsize=20, epochs=10
    )
    
    # X, Y = valset
    # proj_layers = [ProjectionLayer(layer) for layer in layers]
    # proj_layer = proj_layers[0]
    # proj_layer.add_axes(b=ProjectionAxis(arr=proj_layer.layer.b, pos=(7,), num=10, d=10))

    # proj_layer = proj_layers[-1]
    # proj_layer.add_axes(W=ProjectionAxis(arr=proj_layer.layer.W, pos=(7,7), num=5, d=1))
    
    # proj_grid = ProjectionGrid(layers=proj_layers, X=X, Y=Y)
    # proj_grid.compute()
    # print(proj_grid.grid)
    
    training = Training(layers, it, valset, alpha=0.03)
    
    while not training.completed:
        training.train_minibatch()
        sleep(0.1)

    # print("djdw")
    # for djdw in training.training_pipeline.dJdW:
    #     print(djdw)
    # print("output")
    # print(training.training_pipeline.model.A)

main()
