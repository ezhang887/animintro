dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

import sys
sys.path.append("../../")
from torchvggish.torchvggish.vggish import VGGish
# ^ i know this is really gross, but just deal with it for now
# this whole script is a mess of directories, but just run it
# in the scripts directory

import os
import torch

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}


def vggish(**kwargs):
    model = VGGish(urls=model_urls, **kwargs)
    return model

model = vggish()
model.cuda()
model.eval()

audio_dir = '../data/Audio/lofi'
for a in os.listdir(audio_dir):
    print(a)
    if a == 'Made_in_Abyss_13.wav':
        continue
    filename = os.path.join(audio_dir, a)
    with torch.no_grad():
        fp = model.forward(filename)
    savedir = '../data/Audio/vggish_lofi'
    a_no_ext = os.path.splitext(a)[0]
    a_pt = a_no_ext + '.pt'
    savedir = os.path.join(savedir, a_pt)
    torch.save(fp,savedir)
    


# audio_dir = '../data/Audio/lofi'
# a = 'Kimetsu_no_Yaiba_10.wav'
# print(a)
# filename = os.path.join(audio_dir, a)
# print(filename)
# savedir = '../data/Audio/vggish_lofi'
# a_no_ext = os.path.splitext(a)[0]
# a_pt = a_no_ext + '.pt'
# savedir = os.path.join(savedir, a_pt)
# print(savedir)
