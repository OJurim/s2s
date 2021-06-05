
from scipy.io.wavfile import read, write
import os
import torch
import numpy as np
import sys
import librosa

def audio_stretcher(full_path_read, full_path_sing, stretching_factor=0.5, segment_size=3500000,  method="resample"):

    # stretching by stretching_size factor

    from audiotsm import phasevocoder
    from audiotsm.io.wav import WavReader, WavWriter
    with WavReader(full_path_read) as reader:
        with WavWriter('tmp.tmp', reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=stretching_factor)
            tsm.run(reader, writer)
    stretched_sample_rate_read, stretched_audio_read = read('tmp.tmp')
    stretched_sample_rate_sing, stretched_audio_sing = read(full_path_sing)
    os.remove("tmp.tmp")
    # print(len(stretched_audio_read))
    # print(len(stretched_audio_sing))
    if method == "padding":
        # padding with zeros to segment_size length
        padding_size_read = segment_size - len(stretched_audio_read)
        stretched_audio_read = np.pad(stretched_audio_read, (0, padding_size_read), 'constant', constant_values=0)
        padding_size_sing = segment_size - len(stretched_audio_sing)
        stretched_audio_sing = np.pad(stretched_audio_sing, (0, padding_size_sing), 'constant', constant_values=0)
    elif method == "resample":
        # referring segment size as maximal length
        samples_num_read = len(stretched_audio_read)
        dec_factor = samples_num_read/segment_size
        print((stretched_audio_sing[:]))
        stretched_audio_sing = librosa.resample(stretched_audio_sing.astype(np.float), stretched_sample_rate_sing,
                                                stretched_sample_rate_sing*dec_factor)
        stretched_audio_read = librosa.resample(stretched_audio_read.astype(np.float), stretched_sample_rate_read,
                                                stretched_sample_rate_read*dec_factor)

    print(stretched_audio_read, len(stretched_audio_read))
    print(stretched_audio_sing, len(stretched_audio_sing))

    return stretched_audio_read, stretched_sample_rate_read, stretched_audio_sing, stretched_sample_rate_sing




args = sys.argv
#print(args)
max_len = 0
text_path = str(os.path.abspath(args[1])) #.join(args[1])
#print(text_path)
wav_path = str(os.path.abspath(args[2]))
full_stretch_path = wav_path + "/stretched_data/"
if not os.path.exists(full_stretch_path):
    os.mkdir(full_stretch_path)
with open(text_path, 'r') as file_list:
    while True:
        line_read = file_list.readline()
        if not line_read: break
        line_sing = file_list.readline()
        full_path_sing = wav_path + "/" + line_sing[:-1]
        sampling_rate_sing, sing_wav = read(full_path_sing)
        # resampled_sing_wav = librosa.resample(sing_wav, sampling_rate_sing, 16000)
        # write('resampled_audio.wav', int(args[2]), new_audio)
        if len(sing_wav) > max_len:
            max_len = len(sing_wav)
print(max_len)
with open(text_path, 'r') as file_list:
    while True:
        line_read = file_list.readline()
        if not line_read: break
        line_sing = file_list.readline()
        full_path_read = wav_path + "/" + line_read[:-1]
        sampling_rate_read, read_wav = read(full_path_read)
        full_path_sing = wav_path + "/" + line_sing[:-1]
        sampling_rate_sing, sing_wav = read(full_path_sing)
        print(len(sing_wav))
        print(len(read_wav))
        stretching_factor = len(read_wav)/len(sing_wav)
        print(stretching_factor)
        r_wav, r_fs, s_wav, s_fs = audio_stretcher(full_path_read, full_path_sing, stretching_factor, max_len)
        #print(line_read)
        #print(line_sing)
        #print(stretching_factor)
        write(full_stretch_path + line_read[:-1], r_fs, r_wav)
        write(full_stretch_path + line_sing[:-1], s_fs, s_wav)


# def resamp(path_to_wav, target_sr):
#     origin_wav, origin_sr = librosa.load(path_to_wav)
#     print(origin_sr)
#     y = librosa.resample(origin_wav, origin_sr, int(target_sr))
#     return y
#
# args = sys.argv
# new_audio = resamp(args[1], args[2])
# write('resampled_audio.wav', int(args[2]), new_audio)
#
