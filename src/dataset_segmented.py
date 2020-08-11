#!/usr/bin/env python3

import torch
import torchaudio
import os
import numpy as np

from torch.utils.data import Dataset

class AnimeAudioDataset(Dataset):
    """
    Load in audio file names and get the length of the largest audio sample.
    Audio is loaded into memory when _get_item is called
    
    Labels are all loaded into memory and normalized during initialization
    """
    
    def __init__(self):
        self.audio_dir = 'data/Audio/vggish_lofi'
        self.label_dir = 'data/Labels' # change this to whatever
        
        self.audio_filenames = self._get_filenames()
        self.audios = self._load_audio()
        self.labels = self._load_labels()
        self.audios_time = self._append_time()
        self.labels96 = self._segment_labels()
        
        assert len(self.audios) == len(self.labels)
        
        '''
        self.data = self._pad_audio(self._get_filenames())
        self.labels, self.label_mean, self.label_std = \
                self._normalize_labels(self._load_labels())
        '''
    
    def _append_time(self):
        """adds the normalized time stamp as one of the features

        Returns
        -------
        dict (filename : tensor([int, 129]))
            2D matrix same as return of load audio except with an added
            feature that is the normalized timestep of where audio slice
        """
        audios_time = {}
        for filename in self.audios:
            x = self.audios[filename]
            l = x.size()[0]
            
            r = np.arange(l).astype(np.float32)

            # normalize
            r = r/l - .5
            r = np.reshape(r, (l,1))

            # cuda because saved tensors are cuda
            r = torch.from_numpy(r).cuda()
            audios_time[filename] = torch.cat((x,r),1)
        return audios_time


    def _load_audio(self):
        """loads in audio tensors normalized

        Returns
        -------
        dict (filename : tensor([int, 128]))
            2D matrix. Contains 128-len vectors describing each .96 sec slice of 
            the respective audio file. Int would be how many of these slices are
            in the audio file.
        """
        audios = {}
        for filename in self.audio_filenames:
            # load audio
            audio_filename = filename + '.pt'
            audio_path = os.path.join(self.audio_dir, audio_filename)
            audio = torch.load(audio_path)
            
            # normalize
            audio = audio/255-.5
            
            audios[filename] = audio
        return audios


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
        
        for filename in os.listdir(self.label_dir):
            # get only file with no extension
            fe = os.path.splitext(filename)[0]
            if fe == 'Boku_no_Hero_Academia_9' or fe == 'Samurai_Champloo_19':
                # this one is broken
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
        error  = False
        
        for filename in self.audio_filenames:
            label_filename = filename + '.label'
            label_path = os.path.join(self.label_dir, label_filename)
            
            # initialize list for label
            # [start intro in ms, end intro, start outro, end outro]
            label = [-1]*4 
            f = open(label_path, 'r')
            for i in range(4):
                label[i] = int(f.readline())
            labels[filename] = label
        
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
            print (filename)
            y = self.audios[filename]
            l = y.size()[0]
            x = torch.zeros(l)

            label = self.labels[filename]
            start_intro = label[0] // 960 + 1 # round up
            end_intro = label[1] // 960 # round down
            start_outro = label[2] // 960 + 1 # round up
            end_outro = label[3] // 960 # round down

            for i in range(start_intro, end_intro):
                x[i] = 1
            
            for i in range(start_outro, end_outro):
                x[i] = 2

            labels96[filename] = x.cuda()
            
        return labels96
                
    
    def __len__(self):
        return len(self.audio_filenames)
    
    def __getitem__(self, idx):
        # idx can be a tensor
        audio_filename = os.path.join(self.audio_dir, self.audio_filenames[idx])
        waveform, _ = torchaudio.load(audio_filename)
        # padded_audio = self._pad_audio([waveform])
        padded_audio = self._pad_audio(waveform)
        padded_audio = padded_audio.to(self.device)
        return padded_audio, self.labels[idx]