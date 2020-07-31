#!/usr/bin/env python3

import torch
import torchaudio
import os

from torch.utils.data import Dataset

class AnimeAudioDataset(Dataset):
    """
    Load in audio file names and get the length of the largest audio sample.
    Audio is loaded into memory when _get_item is called
    
    Labels are all loaded into memory and normalized during initialization
    """
    
    def __init__(self, device):
        
        self.device = device

        self.audio_dir = 'data/Audio'
        self.label_dir = 'data/Labels/nick'
        
        self.audio_files, self.max_length = self._load_audio()
        raw_labels = self._load_labels(self.audio_files)
        self.labels, self.l_mean, self.l_std = self._normalize_labels(raw_labels)
        self.labels = self.labels.to(self.device)
    
        assert len(self.audio_files) == len(self.labels)
        
        '''
        self.data = self._pad_audio(self._load_audio())
        self.labels, self.label_mean, self.label_std = \
                self._normalize_labels(self._load_labels())
        '''
        
    def _load_audio(self):

        audio_files = []
        max_length = 0
        
        for a in os.listdir(self.audio_dir):
            audio_files.append(a)
            
            audio_file = os.path.join(self.audio_dir, a)
            waveform, _ = torchaudio.load(audio_file)
            max_length = max(max_length, waveform.shape[1])
        
        return audio_files, max_length
    
    def _load_labels(self, audio_files):

        labels = []
        error  = False
        
        for filename in audio_files:
            file = os.path.splitext(filename)[0] + '.label'
            filepath = os.path.join(self.label_dir, file)
            label = [0] * 4
            try:
                f = open(filepath, 'r')
                for i in range(4):
                    label[i] = int(f.readline())
                labels.append(label)
            except BaseException as e:
                print(e)
                error = True
        
        if error:
            raise RuntimeError('Failed to load labels')
            
        return torch.Tensor(labels)
                
    def _normalize_labels(self, labels):
        l_mean = labels.mean(dim=0)
        l_std = labels.std(dim=0)
        labels = (labels - l_mean) / l_std
        return labels, l_mean, l_std
        
    def _pad_audio(self, data):
        # longest = max(map(lambda x: x.shape[1], data))
        '''
        for i in range(len(data)):
            zeros = torch.zeros(2, (self.max_length - data[i].shape[1]))
            data[i] = torch.cat((data[i], zeros), dim=1)
        return torch.stack(data)
        '''
        zeros = torch.zeros(2, (self.max_length - data.shape[1]))
        return torch.cat((data, zeros), dim=1)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # idx can be a tensor
        audio_file = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, _ = torchaudio.load(audio_file)
        # padded_audio = self._pad_audio([waveform])
        padded_audio = self._pad_audio(waveform)
        padded_audio = padded_audio.to(self.device)
        return padded_audio, self.labels[idx]