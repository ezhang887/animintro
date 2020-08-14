import torch

"""
Saves a model to disk. Also saves the optimizer state
so we can use this as a checkpoint to resume training from.
"""


def save_model(filename, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
        "model_state": model.state_dict(),
    }
    with open(filename, "wb") as f:
        torch.save(state, f)


"""
Loads a model from disk to resume training from.

filename - path to where the model is saved
model - nn.Module to load into
optimizer - optimizer to load into to continue training. If None, will only load the weights into the model.
"""


def load_model(filename, model, optimizer = None):
    with open(filename, "rb") as f:
        state = torch.load(filename)

    assert "model_state" in state
    model.load_state_dict(state["model_state"])

    if optimizer is not None:
        assert "optimizer_state" in state
        optimizer.load_state_dict(state["optimizer_state"])
