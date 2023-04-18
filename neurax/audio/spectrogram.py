import numpy as np
import scipy as sp
import torch
import torchaudio

import naplib as nl


def auditory_spectrogram(x, sr):
    mid_sr = 16_000
    
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x = torchaudio.transforms.Resample(sr, mid_sr)(x).squeeze(0).numpy()
    x = nl.features.auditory_spectrogram(x, mid_sr, frame_len=10, tc=10)
    x = np.maximum(x, 0) ** (1/3)
    
    return x


def mel_spectrogram(x, sr):
    mid_sr = 16_000
    mel_len = len(x) * 100 // sr
    mel_config = {
        'sample_rate': mid_sr,
        'n_fft': 1280,
        'win_length': None,
        'hop_length': 320,
        'f_min': 40,
        'f_max': 8000,
        'n_mels': 128,
        'center': True,
        'pad_mode': 'reflect',
        'power': 2.0,
        'norm': 'slaney',
        'mel_scale': 'htk',
    }
    
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x = torchaudio.transforms.Resample(sr, mid_sr)(x)
    x = torchaudio.transforms.MelSpectrogram(**mel_config)(x)
    x = torchaudio.transforms.AmplitudeToDB(top_db=60)(x)
    x = (x - x.min()).squeeze(0).T.numpy()
    x = sp.signal.resample(x, mel_len)
    
    return x

