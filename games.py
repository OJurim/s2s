
from scipy.io.wavfile import read, write
import os
import torch
import numpy as np


def audio_stretcher(file_path, stretching_factor=0.5, segment_size=3500000):
    # stretching by stretching_size factor
    from audiotsm import phasevocoder
    from audiotsm.io.wav import WavReader, WavWriter

    with WavReader(file_path) as reader:
        with WavWriter('tmp.tmp', reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=stretching_factor)
            tsm.run(reader, writer)

    stretched_sample_rate, stretched_audio = read('tmp.tmp')
    os.remove("tmp.tmp")

    # padding with zeros to segment_size length
    stretched_audio = np.pad(stretched_audio, (0, segment_size), 'constant', constant_values=0)
    return stretched_audio, stretched_sample_rate

au, sr = audio_stretcher('./JTAN_sing_15_wav_splitted15.wav', stretching_size=0.5, segment_size=20000)
print(au, sr)