#!/usr/bin/env python3

import torch
import torchaudio
import os

from torch.utils.data import Dataset

from src.utils import pad_audio


class AnimeAudioDataset(Dataset):
    """
    Load in audio file names and get the length of the largest audio sample.
    Audio is loaded into memory when _get_item is called
    
    Labels are all loaded into memory and normalized during initialization
    """
   
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
        
        raw_labels = self._load_labels(self.audio_filenames)
        self.labels, self.l_mean, self.l_std = self._normalize_labels(raw_labels)
        self.labels = self.labels.to(self.device)

        assert len(self.audio_filenames) == len(self.labels)

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
        l_mean = labels.mean(dim=0)
        l_std = labels.std(dim=0)
        labels = (labels - l_mean) / l_std
        return labels, l_mean, l_std

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
