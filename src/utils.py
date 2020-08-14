import torch
import torchaudio
import numpy as np

"""
Saves a model to disk. Also saves the optimizer state
so we can use this as a checkpoint to resume training from.
"""


def save_model(filename, model, optimizer, epoch):
    assert model.mean is not None
    assert model.stddev is not None

    state = {
        "model_state": model.state_dict(),
        "mean": model.mean,
        "stddev": model.stddev,
        "max_audio_len": model.max_length,
        "epoch": epoch,
        "optimizer_state": optimizer.state_dict(),
    }
    with open(filename, "wb") as f:
        torch.save(state, f)


"""
Loads a model from disk to resume training from.

filename - path to where the model is saved
model - nn.Module to load into
optimizer - optimizer to load into to continue training. If None, will only load the weights into the model.

Returns the epoch that training was saved on (or None if the epoch wasn't saved).
"""


def load_model(filename, model, optimizer=None):
    with open(filename, "rb") as f:
        state = torch.load(filename)

    assert "model_state" in state
    model.load_state_dict(state["model_state"])

    assert "mean" in state
    model.mean = state["mean"]

    assert "stddev" in state
    model.stddev = state["stddev"]

    assert "max_audio_len" in state
    model.max_length = state["max_audio_len"]

    if optimizer is not None:
        assert "optimizer_state" in state
        optimizer.load_state_dict(state["optimizer_state"])

    if "epoch" in state:
        return state["epoch"]
    return None


def predict(filename, model, use_cuda=True):
    assert model.mean is not None
    assert model.stddev is not None

    waveform, _ = torchaudio.load(filename)

    waveform = pad_audio(waveform, model.max_length)
    waveform = waveform.unsqueeze(dim=0)

    dummy = waveform.clone()

    for i in range(model.batch_size - 1):
        waveform = torch.cat((waveform, dummy))

    if use_cuda:
        waveform = waveform.cuda()

    prediction = model(waveform)

    prediction = prediction.cpu().reshape(model.batch_size, 4)

    prediction = prediction * model.stddev + model.mean
    return prediction


def pad_audio(data, max_audio_len):
    zeros = torch.zeros(2, (max_audio_len - data.shape[1]))
    return torch.cat((data, zeros), dim=1)
