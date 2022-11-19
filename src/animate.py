from utils.read_params import read_params
from utils.forward_kinematics import (
    revert_coordinate_space,
    _some_variables,
    fkl
)
import matplotlib.pyplot as plt
from utils.viz import Ax3DPose
# from utils.viz import *
from os.path import join
import numpy as np
import logging
import h5py

params = read_params()


def main():
    # Logging
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=20
    )
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()
    filename = "sample.h5"
    filename = join(
        params["results_dir"],
        filename
    )
    with h5py.File(filename, 'r') as h5f:
        # Ground truth (exponential map)
        expmap_gt = h5f['expmap/gt/walking_{}'.format(
            params["sample_id"]
        )][:]
        # Prediction (exponential map)
        expmap_pred = h5f['expmap/preds/walking_{}'.format(
            params["sample_id"])
        ][:]
    # Number of Ground truth/Predicted frames
    nframes_gt, nframes_pred = expmap_gt.shape[0], expmap_pred.shape[0]
    logging.info("{} {}".format(
        nframes_gt,
        nframes_pred)
    )
    # Put them together and revert the coordinate space
    expmap_all = revert_coordinate_space(
        np.vstack(
            (expmap_gt, expmap_pred)),
        np.eye(3),
        np.zeros(3)
    )
    expmap_gt = expmap_all[:nframes_gt, :]
    expmap_pred = expmap_all[nframes_gt:, :]
    # Use forward kinematics to compute 33 3d points for each frame
    xyz_gt = np.zeros((nframes_gt,
                       96))
    xyz_pred = np.zeros((nframes_pred,
                         96))
    for i in range(nframes_gt):
        xyz_gt[i, :] = fkl(
            expmap_gt[i, :],
            parent,
            offset,
            rotInd,
            expmapInd
        )
    for i in range(nframes_pred):
        xyz_pred[i, :] = fkl(
            expmap_pred[i, :],
            parent,
            offset,
            rotInd,
            expmapInd
        )

    # === Plot and animate ===
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ob = Ax3DPose(ax)

    # First, plot the conditioning ground truth
    for i in range(nframes_gt):
        ob.update(
            xyz_gt[i, :],
            lcolor="#ff0000",
            rcolor="#0000ff"
        )
        plt.show(block=False)
        plt.title("Observations")
        fig.canvas.draw()
        plt.pause(0.01)

    # Plot the prediction
    for i in range(nframes_pred):
        ob.update(
            xyz_pred[i, :],
            lcolor="#9b59b6",
            rcolor="#2ecc71"
        )
        plt.show(block=False)
        plt.title("Predictions")
        fig.canvas.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    main()
