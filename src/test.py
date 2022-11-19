
"""Simple code for training an RNN for motion prediction."""
from utils.evaluation import evaluate_batch
from utils.read_params import read_params
from utils.data_utils import (
    revert_output_format,
    unNormalizeData,
    define_actions,
    read_all_data,
    expmap2rotmat,
    rotmat2euler,
)
from pandas import DataFrame
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
import h5py

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

device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "cpu")
makedirs(train_dir,
         exist_ok=True)


def get_srnn_gts(actions,
                 model,
                 test_set,
                 data_mean,
                 data_std,
                 dim_to_ignore,
                 to_euler=True) -> None:
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles
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
            the values are the ground_truth, denormalized expected
            outputs of srnns's seeds.
    """
    srnn_gts_euler = {}
    for action in actions:
        srnn_gt_euler = []
        # get_batch or get_batch_srnn
        _, _, srnn_expmap = model.get_batch_srnn(
            test_set,
            action,
            5,
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
                            expmap2rotmat(denormed[j, k:k+3]
                                          ))
            srnn_gt_euler.append(denormed)

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler
    return srnn_gts_euler


def main():
    """Sample predictions for srnn's seeds"""
    actions = define_actions(params["action"])
    nsamples = 8
    # === Create the model ===
    logging.info("Creating a model with {} units.".format(params['size']))
    logging.info("Loading model")
    model_name = f"model_{params['iterations']}"
    model_name = join(
        train_dir,
        model_name
    )
    model = torch.load(model_name)
    model.source_seq_len = params["seq_length_in"]
    model.target_seq_len = 100
    model = model.to(device)
    logging.info("Model created")
    # Load all the data
    _, test_set, data_mean, data_std, dim_to_ignore, _ = read_all_data(
        actions,
        50,
        params["seq_length_in"],
        params["data_dir"],
    )
    # === Read and denormalize the gt with srnn's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    srnn_gts_expmap = get_srnn_gts(
        actions,
        model,
        test_set,
        data_mean,
        data_std,
        dim_to_ignore,
        to_euler=False
    )
    srnn_gts_euler = get_srnn_gts(
        actions,
        model,
        test_set,
        data_mean,
        data_std,
        dim_to_ignore
    )
    # Clean and create a new h5 file of samples
    SAMPLES_FNAME = "sample.h5"
    SAMPLES_FNAME = join(
        params["results_dir"],
        SAMPLES_FNAME
    )
    try:
        remove(SAMPLES_FNAME)
    except OSError:
        pass

    results = DataFrame()
    # Predict and save for each action
    for action in actions:

        # Make prediction with srnn' seeds
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_srnn(
            test_set,
            action,
            5,
            device
        )
        # Forward pass
        srnn_poses = model(
            encoder_inputs,
            decoder_inputs,
            device
        )
        srnn_loss = (srnn_poses - decoder_outputs)**2
        srnn_loss.cpu().data.numpy()
        srnn_loss = srnn_loss.mean()
        srnn_poses = srnn_poses.cpu().data.numpy()
        srnn_poses = srnn_poses.transpose([1, 0, 2])
        srnn_loss = srnn_loss.cpu().data.numpy()
        # Restores the data in the same format as the original: dimension 99.
        # Returns a tensor of size (batch_size, seq_length, dim) output.
        srnn_pred_expmap = revert_output_format(
            srnn_poses, data_mean, data_std, dim_to_ignore, actions)
        # Save the samples
        with h5py.File(SAMPLES_FNAME, 'a') as hf:
            for i in range(nsamples):
                # Save conditioning ground truth
                node_name = 'expmap/gt/{1}_{0}'.format(i, action)
                hf.create_dataset(node_name, data=srnn_gts_expmap[action][i])
                # Save prediction
                node_name = 'expmap/preds/{1}_{0}'.format(i, action)
                hf.create_dataset(node_name, data=srnn_pred_expmap[i])

        # Compute and save the errors here
        mean_errors_batch = evaluate_batch(
            srnn_pred_expmap,
            srnn_gts_euler[action]
        )
        results[action] = mean_errors_batch
        # text = 'Mean error for test data'
        # text1 = '{} along the horizon on action {}: {}'.format(
        # text,
        # action,
        # mean_errors_batch
        # )
        # text2 = '{} at horizon {} on action {}: {}'.format(
        # text,
        # params["horizon_test_step"],
        # action,
        # mean_errors_batch[params["horizon_test_step"]]
        # )
        # logging.info(text1)
        # logging.info(text2)
        with h5py.File(SAMPLES_FNAME, 'a') as hf:
            node_name = 'mean_{0}_error'.format(action)
            hf.create_dataset(
                node_name,
                data=mean_errors_batch
            )
    mean = DataFrame(results.mean())
    mean.index.name = "Action"
    mean.columns = ["Mean error"]
    filename = "mean_error.csv"
    filename = join(
        params["results_dir"],
        filename
    )
    results.to_csv(filename,
                   index=False)
    filename = "mean_batch_error.csv"
    filename = join(
        params["results_dir"],
        filename
    )
    mean.to_csv(filename)
    return


if __name__ == "__main__":
    main()
