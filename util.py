import torch
import numpy as np
import IPython.display as ipd
import typing as tp
import librosa


def display_audio(samples: tp.List[torch.Tensor], path: str = None):
    audio = np.concatenate(samples, axis=0)
    audio = np.squeeze(audio)
    ipd.display(ipd.Audio(audio, rate=32000))

    if path:
        librosa.output.write_wav(path, audio, 32000)
