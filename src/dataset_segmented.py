#!/usr/bin/env python3

import torch
import torchaudio
import os
import numpy as np

from torch.utils.data import Dataset


class AnimeAudioDataset(Dataset):
    """
    Loads all audio file data. Includes:
    features, labels, time, labels96
    """

    # TODO: add normalized flag
    def __init__(self, l96=True):
        self.feature_dir = "data/Audio/vggishLofi/"
        self.label_dir = "data/Labels"  # change this to whatever

        self.audio_filenames = self._get_filenames()
        self.features = self._load_audio()
        self.labels = self._load_labels()
        # self.time = self._load_time()
        self.labels96 = self._segment_labels()

        self.l96 = l96

        assert len(self.features) == len(self.labels)

    def _load_time(self):
        """adds the normalized time stamp as one of the features

        Returns
        -------
        dict (filename : tensor([int, 1]))
            2D matrix(1D vertical to match features so that 
                      torch.cat((features[i],audios_time[i]),1) works)
            feature that is the normalized timestep of where audio slice
        """
        time = {}
        for filename in self.features:
            x = self.features[filename]
            l = x.size()[0]

            r = np.arange(l).astype(np.float32)

            # normalize TODO: take in normalized flag
            r = r / l - 0.5
            r = np.reshape(r, (l, 1))

            # cuda because saved tensors are cuda
            r = torch.from_numpy(r).cuda()
            time[filename] = r
        return time

    def _load_audio(self):
        """loads in audio tensors normalized

        Returns
        -------
        dict (filename : tensor([int, 128]))
            2D matrix. Contains 128-len vectors describing each .96 sec slice of 
            the respective audio file. Int would be how many of these slices are
            in the audio file.
        dict (filename : tensor([int, 1, 96, 64])
            basicalky 3d mat. contains 96x64 matricies representing the log 
            mel spectrogram of the audio. Int is same as above
        """
        features = {}

        # for filename in self.audio_filenames:
        for filename in self.audio_filenames:
            # load audio
            # filename += ".pt"

            feature_path = os.path.join(self.feature_dir, filename + ".pt")
            feature = torch.load(feature_path)

            features[filename] = feature

        return features

    def _get_filenames(self):
        """loads in audio filenames without extension

        Returns
        -------
        [str, ...]
            array of filenames with no extension
            'episode_1' instead of 'episode_1.wav' or 'episode_1.label'
        """
        audio_filenames = []
        max_length = 0

        for filename in os.listdir(self.feature_dir):
            # get only file with no extension
            fe = os.path.splitext(filename)[0]
            if (
                fe == "Boku_no_Hero_Academia_9"
                or fe == "Samurai_Champloo_19"
                or fe == "Haikyuu!!_9"
            ):
                # these are broken in someway
                continue
            audio_filenames.append(fe)

        return audio_filenames

    def _load_labels(self):
        """loads the labels corresponding to the audio filenames

        Parameters
        ----------
        audio_filenames : [[str, ...]
            list of filenames

        Returns
        -------
        dict(filename : label)
            label : [start intro in ms, end intro, start outro, end outro]
        """
        labels = {}
        error = False

        missing_labels = []

        for filename in self.audio_filenames:
            label_filename = filename + ".label"

            label_path = os.path.join(self.label_dir, label_filename)

            # initialize list for label
            # [start intro in ms, end intro, start outro, end outro]
            label = [-1] * 4
            try:
                f = open(label_path, "r")
            except:
                missing_labels.append(label_filename)
                continue

            for i in range(4):
                label[i] = int(f.readline())
            labels[filename] = label

        if len(missing_labels):
            raise RuntimeError(f"Missing Labels: {missing_labels}")

        return labels

    def _segment_labels(self):
        """divide in 960 ms chunks label outtro intro

        Returns
        -------
        dict(filename : tensor([int, ...])
            tensor is 1d array of 0,1,2.
            Each index represents a consective 960 ms(.96 s) audio slice.
            1 is intro, 2 is outro, 0 is neither
        """
        labels96 = {}
        for filename in self.labels:
            y = self.features[filename]
            l = y.size()[0]

            x = torch.zeros(l)

            label = self.labels[filename]
            start_intro = label[0] // 960 + 1  # round up
            end_intro = label[1] // 960  # round down
            start_outro = label[2] // 960 + 1  # round up
            end_outro = label[3] // 960  # round down

            for i in range(start_intro, end_intro):
                x[i] = 1

            for i in range(start_outro, end_outro):
                x[i] = 2

            labels96[filename] = x.cuda()

        return labels96

    def __len__(self):
        return len(self.audio_filenames)

    def get_all(self, key):
        return self.features[key], self.labels[key], self.time[key], self.labels96[key]

    def __getitem__(self, idx):
        # TODO: update this
        # for now make this return audios instead of audios_time

        # idx can be a tensor
        key = self.audio_filenames[idx]
        if self.l96:
            return self.features[key], self.labels96[key]
        return self.features[key], self.labels[key]
