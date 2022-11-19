import moviepy.video.io.ImageSequenceClip as MovieMaker
from utils.read_params import read_params
from utils.forward_kinematics import (
    revert_coordinate_space,
    _some_variables,
    fkl
)
import matplotlib.pyplot as plt
from utils.viz import Ax3DPose
from os.path import join
from numpy import (
    zeros,
    vstack,
    eye,
)
import logging
import h5py

params = read_params()


def create_animation(filenames: list,
                     name: str = "Animation",
                     fps: int = 20) -> None:
    """
    Funcion que ejecuta la creacion de la animacion
    """
    output_file = f"{name}.mp4"
    movie = MovieMaker.ImageSequenceClip(
        filenames,
        fps=fps
    )
    movie.write_videofile(output_file,
                          logger=None)


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
        vstack(
            (expmap_gt, expmap_pred)),
        eye(3),
        zeros(3)
    )
    expmap_gt = expmap_all[:nframes_gt, :]
    expmap_pred = expmap_all[nframes_gt:, :]
    # Use forward kinematics to compute 33 3d points for each frame
    xyz_gt = zeros((nframes_gt,
                    96))
    xyz_pred = zeros((nframes_pred,
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
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ob1 = Ax3DPose(ax1)
    ob2 = Ax3DPose(ax2)
    # First, plot the conditioning ground truth
    n_len = len(str(nframes_gt))
    filenames = list()
    for i in range(nframes_gt):
        ob1.update(
            xyz_gt[i, :],
            lcolor="#ff0000",
            rcolor="#0000ff",
            title="Observations",
        )
        # ob1.set_title("Observations")
        ob2.update(
            xyz_pred[i, :],
            lcolor="#9b59b6",
            rcolor="#2ecc71",
            title="Predicted",
        )
        plt.tight_layout()
        filename = f"{i}"
        filename = filename.zfill(n_len)
        filename = f"{filename}.png"
        filename = join(
            params["graphics_dir"],
            filename
        )
        plt.savefig(filename)
        filenames.append(filename)
    create_animation(
        filenames
    )


if __name__ == '__main__':
    main()
