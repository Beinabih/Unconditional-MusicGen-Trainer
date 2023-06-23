import numpy as np
import IPython.display as ipd

def display_audio(samples, path=None):
    audio = np.concatenate(samples, axis=0)
    audio = np.squeeze(audio)
    ipd.display(ipd.Audio(audio, rate=32000))

    if path:
        librosa.output.write_wav(path, audio, 32000)