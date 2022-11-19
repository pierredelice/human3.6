
"""Simple code for training an RNN for motion prediction."""
from models.motionpredictor import MotionPredictor
from utils.read_params import read_params
from utils.data_utils import (
    define_actions,
    read_all_data,
)
from pandas import DataFrame
from torch.optim import Adam
from os import makedirs
from numpy import array
from os.path import (
    normpath,
    join
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
if params["log_file_train"] == "":
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=params["log_level"]
    )
else:
    logging.basicConfig(
        filename=params["log_file_train"],
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

logging.info(f"Train dir: {train_dir}")
makedirs(train_dir,
         exist_ok=True)


def main():
    """Train a seq2seq model on human motion"""
    # Set of actions
    actions = define_actions(params["action"])
    number_of_actions = len(actions)

    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
        actions,
        params["seq_length_in"],
        params["seq_length_out"],
        params["data_dir"],
    )

    # Create model for training only
    model = MotionPredictor(
        params["seq_length_in"],
        params["seq_length_out"],
        params["size"],
        params["batch_size"],
        params["learning_rate"],
        params["learning_rate_decay_factor"],
        number_of_actions
    )
    model = model.to(device)
    # This is the training loop
    loss, val_loss = 0.0, 0.0
    all_val_losses = []
    current_step = 0
    all_losses = []
    # The optimizer
    optimiser = Adam(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(0.9, 0.999)
    )
    iterations = int(params["iterations"])
    for _ in range(iterations):
        optimiser.zero_grad()
        # Set a flag to compute gradients
        model.train()
        # === Training step ===
        # Get batch from the training set
        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
            train_set,
            actions,
            device
        )
        # Forward pass
        preds = model(
            encoder_inputs,
            decoder_inputs,
            device
        )
        # Loss: Mean Squared Errors
        step_loss = (preds-decoder_outputs)**2
        step_loss = step_loss.mean()
        # Backpropagation
        step_loss.backward()
        # Gradient descent step
        optimiser.step()
        step_loss = step_loss.cpu().data.numpy()
        if current_step % 10 == 0:
            logging.info("step {0:04d}; step_loss: {1:.4f}".format(
                current_step,
                step_loss
            ))
        loss += step_loss / params["test_every"]
        current_step += 1
        # === step decay ===
        if current_step % params["learning_rate_step"] == 0:
            params["learning_rate"] *= params["learning_rate_decay_factor"]
            optimiser = Adam(
                model.parameters(),
                lr=params["learning_rate"],
                betas=(0.9, 0.999)
            )
            print("Decay learning rate. New value at".format(
                params["learning_rate"],
            ))
        # Once in a while, save checkpoint, print statistics.
        if current_step % params["test_every"] == 0:
            model.eval()
            # === Validation ===
            encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch(
                test_set,
                actions,
                device
            )
            preds = model(
                encoder_inputs,
                decoder_inputs,
                device
            )
            step_loss = (preds-decoder_outputs)**2
            val_loss = step_loss.mean()
            print("\n============================\n"
                  "Global step:         %d\n"
                  "Learning rate:       %.4f\n"
                  "Train loss avg:      %.4f\n"
                  "--------------------------\n"
                  "Val loss:            %.4f\n"
                  "============================\n" % (
                      current_step,
                      params["learning_rate"],
                      loss, val_loss)
                  )
            all_val_losses.append(
                [current_step,
                 val_loss.cpu().detach().numpy()]
            )
            all_losses.append([
                current_step,
                loss]
            )
            model_name = f"model_{current_step}"
            model_name = join(
                train_dir,
                model_name
            )
            torch.save(model,
                       model_name)
            # Reset loss
            loss = 0
    vlosses = array(all_val_losses)
    tlosses = array(all_losses)
    losses = DataFrame()
    losses["Validation"] = vlosses[:, 1]
    losses["Total"] = tlosses[:, 1]
    losses.to_csv("test.csv")


if __name__ == "__main__":
    main()
