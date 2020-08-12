#!/usr/bin/env python3

import random
import torch
import torchaudio
import os

from torch.utils.data import Dataset

def load_audio_filenames(audio_dir):
    audio_filenames = []
    max_length = 0

    for a in os.listdir(audio_dir):
        audio_filenames.append(a)
        audio_filename = os.path.join(audio_dir, a)
        waveform, _ = torchaudio.load(audio_filename)

        max_length = max(max_length, waveform.shape[1])

    return audio_filenames, max_length

def get_train_val_data(audio_dir, label_dir, device, validation_split=0.2, load_all_in_mem=False):
    
    audio_filenames, max_length = load_audio_filenames(audio_dir)
    random.shuffle(audio_filenames)

    split_num = int(len(audio_filenames) * validation_split)

    train_filenames = audio_filenames[split_num:]
    val_filenames = audio_filenames[:split_num]

    train_dataset = AudioDataset(train_filenames, 
                                audio_dir,
                                label_dir,
                                max_length,
                                device,
                                load_all_in_mem)
    val_dataset = AudioDataset(val_filenames,
                                audio_dir,
                                label_dir,
                                max_length,
                                device,
                                load_all_in_mem,
                                l_mean=train_dataset.l_mean,
                                l_std=train_dataset.l_std)

    return train_dataset, val_dataset

class AudioDataset(Dataset):
    """
    Load in audio file names and get the length of the largest audio sample.
    Audio is loaded into memory when _get_item is called
    
    Labels are all loaded into memory and normalized during initialization
    """

    def __init__(self,
            audio_filenames,
            audio_dir,
            label_dir,
            max_length,
            device,
            load_all_in_mem=False,
            l_mean=None,
            l_std=None):

        self.audio_filenames = audio_filenames
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.max_length = max_length
        self.device = device
        self.load_all = load_all_in_mem
        self.l_mean = l_mean
        self.l_std = l_std
        
        self.raw_labels = self._load_labels(self.audio_filenames)
        self.labels, self.l_mean, self.l_std = self._normalize_labels(self.raw_labels)
        self.labels = self.labels.to(self.device)

    '''
    def __init__(self, audio_dir, label_dir, device, load_all_in_mem=False):
        
        self.load_all = load_all_in_mem
        self.device = device

        self.audio_dir = audio_dir
        self.label_dir = label_dir

        if self.load_all:
            self.audio_filenames, self.max_length, self.audio_data = self._load_audio()
            for i in range(len(self.audio_data)):
                self.audio_data[i] = self._pad_audio(self.audio_data[i])
            self.audio_data = torch.cat(self.audio_data)
            self.audio_data.to(self.device)
        else:
            self.audio_filenames, self.max_length, _ = self._load_audio()
        
        self.raw_labels = self._load_labels(self.audio_filenames)
        self.labels, self.l_mean, self.l_std = self._normalize_labels(self.raw_labels)
        self.labels = self.labels.to(self.device)

        assert len(self.audio_filenames) == len(self.labels)
    '''

    def _load_audio(self):
        """loads in audio filenames and finds maximum length of all audio segments

        Returns
        -------
        [str, ...], int
            array of filenames, maximum length of all audio segments
        """
        audio_data = [] # Used only if we can load all the data into memory
        audio_filenames = []
        max_length = 0

        for a in os.listdir(self.audio_dir):
            audio_filenames.append(a)
            audio_filename = os.path.join(self.audio_dir, a)
            waveform, _ = torchaudio.load(audio_filename)

            if self.load_all:
                audio_data.append(waveform)
            max_length = max(max_length, waveform.shape[1])
        return audio_filenames, max_length, audio_data
    
    def _load_labels(self, audio_filenames):
        """loads the labels corresponding to the audio filenames

        Parameters
        ----------
        audio_filenames : [[str, ...]
            list of filenames

        Returns
        -------
        tensor([[float,float,float,float], ...])
            array of floats is [intro_start, intro_end, outro_start, outro_end]
            len(audio_filenames) x 4 size tensor of all the labels in milliseconds

        Raises
        ------
        RuntimeError
            If labels fail to load then error is logged, 
            but attempts to finish and raises error after
        """
        labels = []
        error = False

        for filename in audio_filenames:
            file = os.path.splitext(filename)[0] + ".label"
            filepath = os.path.join(self.label_dir, file)
            label = [0] * 4
            try:
                f = open(filepath, "r")
                for i in range(4):
                    label[i] = int(f.readline())
                labels.append(label)
            except BaseException as e:
                print(e)
                error = True

        if error:
            raise RuntimeError("Failed to load labels")

        return torch.Tensor(labels)

    def _normalize_labels(self, labels):
        """normalizes labels off entire mean and std
        returns labels, mean, std
        """
        if self.l_mean is not None and self.l_std is not None:
            labels = (labels - self.l_mean) / self.l_std
            return labels, self.l_mean, self.l_std
        l_mean = labels.mean(dim=0)
        l_std = labels.std(dim=0)
        labels = (labels - l_mean) / l_std
        return labels, l_mean, l_std

    def _pad_audio(self, data):
        # longest = max(map(lambda x: x.shape[1], data))
        """
        for i in range(len(data)):
            zeros = torch.zeros(2, (self.max_length - data[i].shape[1]))
            data[i] = torch.cat((data[i], zeros), dim=1)
        return torch.stack(data)
        """
        zeros = torch.zeros(2, (self.max_length - data.shape[1]))
        return torch.cat((data, zeros), dim=1)

    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
       # idx can be a tensor'
        if self.load_all:
            print("loading from gpu mem")
            return self.audio_data[idx], self.labels[idx]
        else:
            audio_filename = os.path.join(self.audio_dir, self.audio_filenames[idx])
            waveform, _ = torchaudio.load(audio_filename)
            
            padded_audio = self._pad_audio(waveform)
            padded_audio = padded_audio.to(self.device)
            return padded_audio, self.labels[idx]
