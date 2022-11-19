
"""Simple code for training an RNN for motion prediction."""
from utils.evaluation import evaluate_batch
from utils.read_params import read_params
from utils.data_utils import (
    revert_output_format,
    unNormalizeData,
    read_all_data,
    expmap2rotmat,
    rotmat2euler,
)
# from utils.data_utils import *
from numpy import finfo, ones
from os.path import (
    normpath,
    join
)
from os import (
    makedirs,
    remove,
)
import logging
import torch

params = read_params()
train_dir = join(
    params["train_dir"],
    params["action"],
    f"out_{params['seq_length_out']}",
    f"iterations_{params['iterations']}",
    f"size_{params['size']}",
    f"lr_{params['learning_rate']}"
)
train_dir = normpath(train_dir)

# Logging
if params["log_file_test"] == "":
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=params["log_level"]
    )
else:
    logging.basicConfig(
        filename=params["log_file_test"],
        format='%(levelname)s: %(message)s',
        level=params["log_level"],
    )
# Logging
if params["log_file_test"] == "":
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=params["log_level"]
    )
else:
    logging.basicConfig(
        filename=params["log_file_test"],
        format='%(levelname)s: %(message)s',
        level=params["log_level"],
    )
# Detect device
if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    logging.info("cpu")
device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "cpu")
makedirs(
    train_dir,
    exist_ok=True
)


def get_srnn_gts(actions,
                 model,
                 test_set,
                 subject,
                 data_mean,
                 data_std,
                 dim_to_ignore,
                 to_euler=True):
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
        (the error is always computed in Euler angles).

    Args
            actions: a list of actions to get ground truths for.
            model: training model we are using (we only use the
                        "get_batch" method).
            test_set: dictionary with normalized training data.
            data_mean: d-long vector with the mean of the training data.
            data_std: d-long vector with the standard deviation of the
                        training data.
            dim_to_ignore: dimensions that we are not using to train/predict.
            to_euler: whether to convert the angles to Euler format or
                        keep thm in exponential map

    Returns
            srnn_gts_euler: a dictionary where the keys are actions, and
                        the values
            are the ground_truth, denormalized expected outputs of
                        srnns's seeds.
    """
    srnn_gts_euler = {}
    for action in actions:
        srnn_gt_euler = []
        # get_batch or get_batch_srnn
        _, _, srnn_expmap = model.get_batch_srnn(
            test_set,
            action,
            subject,
            device
        )
        srnn_expmap = srnn_expmap.cpu()
        # expmap -> rotmat -> euler
        for i in range(srnn_expmap.shape[0]):
            denormed = unNormalizeData(
                srnn_expmap[i, :, :],
                data_mean,
                data_std,
                dim_to_ignore,
                actions
            )
            if to_euler:
                for j in range(denormed.shape[0]):
                    for k in range(3, 97, 3):
                        denormed[j, k:k + 3] = rotmat2euler(
                            expmap2rotmat(
                                denormed[j, k:k+3]
                            ))
            srnn_gt_euler.append(denormed)
        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler
    return srnn_gts_euler


def main():
    actions = ["walking"]
    test_subject = 8
    nsamples = 20
    generative = False
    # === Create the model ===
    logging.info("Creating a model with {} units.".format(params["size"]))
    sampling = True
    logging.info("Loading model")
    model_name = f"model_4800"
    model_name = join(
        train_dir,
        model_name
    )
    model = torch.load(model_name)
    model.source_seq_len = 50
    model.target_seq_len = 100
    model = model.to(device)
    logging.info("Model created")

    # Load all the data
    _, test_set, data_mean, data_std, dim_to_ignore, _ = read_all_data(
        actions,
        50,
        params["seq_length_out"],
        params["data_dir"],
    )
    print(test_set)
    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts(
        actions,
        model,
        test_set,
        test_subject,
        data_mean,
        data_std,
        dim_to_ignore,
        to_euler=False
    )
    srnn_gts_euler = get_srnn_gts(
        actions,
        model,
        test_set,
        test_subject,
        data_mean,
        data_std,
        dim_to_ignore
    )

    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = 'samples.h5'
    try:
        remove(SAMPLES_FNAME)
    except OSError:
        pass
    action = 'walking'
    logging.info("Action {}".format(action))
    # Make prediction with srnn' seeds
    encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(
        test_set,
        action,
        test_subject,
        device
    )
    # Forward pass
    if not generative:
        # Deterministic model
        srnn_poses = model(
            encoder_inputs,
            decoder_inputs,
            device
        )
        srnn_poses = srnn_poses.unsqueeze(1).repeat(1, nsamples, 1, 1)
    else:
        # Generative model
        # Output should be batch_size x nsamples x target_seq_len x 55
        srnn_poses = model(
            encoder_inputs,
            decoder_inputs,
            device
        )
    srnn_poses = srnn_poses.cpu().data.numpy()
    min_mean_errors_batch = finfo(
        srnn_poses.dtype).max*ones((srnn_poses.shape[0]))
    # Cycling over the batch elements
    for t in range(srnn_poses.shape[0]):
        for j in range(nsamples):
            srnn_poses_j = srnn_poses[t:t+1, j].transpose([1, 0, 2])
            # Restores the data in the same format as the original:
            # dimension 99.
            # Returns a tensor of size (batch_size, seq_length, dim) output.
            srnn_pred_expmap = revert_output_format(
                srnn_poses_j,
                data_mean,
                data_std,
                dim_to_ignore,
                actions
            )
            # Compute and save the errors here
            mean_errors_batch = evaluate_batch(
                srnn_pred_expmap,
                srnn_gts_euler[action]
            )
            mean_errors_batch = mean_errors_batch.mean()
            if mean_errors_batch < min_mean_errors_batch[t]:
                min_mean_errors_batch[t] = mean_errors_batch

    logging.info('Your result on the {} action: {:.3f}'.format(
        action,  min_mean_errors_batch.mean()))
    return


if __name__ == "__main__":
    main()
