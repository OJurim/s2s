
from scipy.io.wavfile import read, write
import soundfile as sf
import os
import torch
import numpy as np
import sys
import librosa
import resize_right
import interp_methods
import matplotlib.pyplot as plt


def audio_stretcher(full_path_read, full_path_sing, stretching_factor=0.5, segment_size=3500000,  method="resample"):

    # stretching by stretching_size factor
    from audiotsm import phasevocoder
    from audiotsm.io.wav import WavReader, WavWriter
    with WavReader(full_path_read) as reader:
        with WavWriter('tmp.tmp', reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=stretching_factor)
            tsm.run(reader, writer)
    # stretched_sample_rate_read, stretched_audio_read = read('tmp.tmp')
    # stretched_sample_rate_sing, stretched_audio_sing = read(full_path_sing)
    # os.remove("tmp.tmp")
    # print(f'read TSM: {len(stretched_audio_read)}')
    # print(f'sing TSM: {len(stretched_audio_sing)}')
    # sf.write(f'./stret/{full_path_read.split("/")[-1]}', stretched_audio_read, stretched_sample_rate_read)
    # sf.write(f'./stret/{full_path_sing.split("/")[-1]}', stretched_audio_sing, stretched_sample_rate_sing)
    stretched_audio_read, stretched_sample_rate_read = sf.read('tmp.tmp')

    # # saves read wav after tsm
    # sf.write("tsm_out_tmp.wav", stretched_audio_read, stretched_sample_rate_read)
    # os.remove("tmp.tmp")
    # sys.exit()


    stretched_audio_sing, stretched_sample_rate_sing = sf.read(full_path_sing)
    os.remove("tmp.tmp")

    if method == "padding":
        # padding with zeros to segment_size length
        padding_size_read = segment_size - len(stretched_audio_read)
        stretched_audio_read = np.pad(stretched_audio_read, (0, padding_size_read), 'constant', constant_values=0)
        padding_size_sing = segment_size - len(stretched_audio_sing)
        stretched_audio_sing = np.pad(stretched_audio_sing, (0, padding_size_sing), 'constant', constant_values=0)
    elif method == "resample":
        long_wav = max(len(stretched_audio_read), len(stretched_audio_sing))
        short_wav = min(len(stretched_audio_read), len(stretched_audio_sing))
        padding_size = long_wav - short_wav
        if len(stretched_audio_read) > len(stretched_audio_sing):
            stretched_audio_sing = np.pad(stretched_audio_sing, (0, padding_size), 'constant', constant_values=0)
        else:
            stretched_audio_read = np.pad(stretched_audio_read, (0, padding_size), 'constant', constant_values=0)
        # print(f'read Padding: {len(stretched_audio_read)}')
        # print(f'sing Padding: {len(stretched_audio_sing)}')
        # write("read_after_pad.wav", stretched_sample_rate_read, stretched_audio_read)
        # write("sing_after_pad.wav", stretched_sample_rate_sing, stretched_audio_sing)
        # sf.write(f'./pad/{full_path_read.split("/")[-1]}', stretched_audio_read, 22050)
        # sf.write(f'./pad/{full_path_sing.split("/")[-1]}', stretched_audio_sing, 22050)
        # referring segment size as maximal length
        samples_num = len(stretched_audio_read)
        dec_factor = segment_size/samples_num
        # print((stretched_audio_sing[:]))
        # print(f'before:{stretched_audio_sing}')
        # print(f'before:{stretched_audio_read}')
        stretched_audio_sing = resize_right.resize(torch.Tensor(stretched_audio_sing), out_shape=(segment_size,))
        stretched_audio_read = resize_right.resize(torch.Tensor(stretched_audio_read), scale_factors=None, out_shape=(segment_size,))
        # print(f'after:{stretched_audio_sing}')
        # print(f'after:{stretched_audio_read}')
        # stretched_audio_sing = librosa.resample(stretched_audio_sing.astype(np.float), stretched_sample_rate_sing,
                                               # stretched_sample_rate_sing*dec_factor)
        # stretched_audio_read = librosa.resample(stretched_audio_read.astype(np.float), stretched_sample_rate_read,
                                               # stretched_sample_rate_read*dec_factor)

    # print(stretched_audio_read, len(stretched_audio_read))
    # print(stretched_audio_sing, len(stretched_audio_sing))
    # print(f'read after: {len(stretched_audio_read)}')
    # print(f'sing after: {len(stretched_audio_sing)}')
    # print(stretched_sample_rate_read*dec_factor)
    read_fs = int(stretched_sample_rate_read*dec_factor)
    sing_fs = int(stretched_sample_rate_sing*dec_factor)

    # sf.write(f'{full_path_read.split("/")[-1]}', stretched_audio_read, samplerate=read_fs, subtype='PCM_16')
    # sf.write(f'{full_path_sing.split("/")[-1]}', stretched_audio_sing, samplerate=sing_fs, subtype='PCM_16')
    return stretched_audio_read, read_fs, stretched_audio_sing, sing_fs

def preprocess():
    args = sys.argv
    #print(args)
    max_len = 0
    text_path = str(os.path.abspath(args[1])) #.join(args[1])
    #print(text_path)
    wav_path = str(os.path.abspath(args[2]))
    full_stretch_path = wav_path + "/stretched_data/"
    wav_path_22050 = wav_path + "/splitted_22050/"
    full_stretch_resampled_path = wav_path + "/stretched_data_22050/"
    if not os.path.exists(full_stretch_path):
        os.mkdir(full_stretch_path)

    # if not os.path.exists(wav_path_22050):
    #     os.mkdir(wav_path_22050)
    with open(text_path, 'r') as file_list:
        while True:
            line_read = file_list.readline()
            if not line_read: break
            # line_sing = file_list.readline()
            line_sing = line_read.replace("read", "sing")
            full_path_sing = wav_path + "/" + line_sing[:-1]
            full_path_read = wav_path + "/" + line_read[:-1]
            sing_wav, sampling_rate_sing = sf.read(full_path_sing)
            # read_wav, sampling_rate_read = sf.read(full_path_read)
            # print("song sr: {}".format(sampling_rate_sing))
            # print("read sr: {}".format(sampling_rate_read))
            # scale_factor = 22050/48000
            # resampled_audio_sing = resize_right.resize(torch.Tensor(sing_wav), out_shape=(len(sing_wav)*scale_factor,))
            # resampled_audio_read = resize_right.resize(torch.Tensor(read_wav), out_shape=(len(read_wav)*scale_factor,))

            # resampled_sing_wav = librosa.resample(sing_wav, sampling_rate_sing, 16000)
            # sf.write(wav_path_22050 + line_sing[:-1], resampled_audio_sing, 22050)
            # sf.write(wav_path_22050 + line_read[:-1], resampled_audio_read, 22050)
            if len(sing_wav) > max_len:
                max_len = len(sing_wav)


    print(f'max_len is {max_len}')
    with open(text_path, 'r') as file_list:
        while True:
            line_read = file_list.readline()
            if not line_read: break
            # line_sing = file_list.readline()
            line_sing = line_read.replace("read", "sing")
            full_path_read = wav_path + "/" + line_read[:-1]
            sampling_rate_read, read_wav = read(full_path_read)
            full_path_sing = wav_path + "/" + line_sing[:-1]
            sampling_rate_sing, sing_wav = read(full_path_sing)
            # print(len(sing_wav))
            # print(len(read_wav))
            stretching_factor = len(read_wav)/len(sing_wav)
            # print(stretching_factor)
            r_wav, r_fs, s_wav, s_fs = audio_stretcher(full_path_read, full_path_sing, stretching_factor=stretching_factor, segment_size=max_len)
            #print(line_read)
            # print(line_sing)
            #print(stretching_factor)
            sf.write(full_stretch_path + line_sing[:-1], s_wav, s_fs)
            sf.write(full_stretch_path + line_read[:-1], r_wav, r_fs)


    # def resamp(path_to_wav, target_sr):
    #     origin_wav, origin_sr = librosa.load(path_to_wav)
    #     print(origin_sr)
    #     y = librosa.resample(origin_wav, origin_sr, int(target_sr))
    #     return y
    #
    # args = sys.argv
    # new_audio = resamp(args[1], args[2])
    # write('resampled_audio.wav', int(args[2]), new_audio)

    # def test(read_path, sing_path):
    #     sampling_rate_read, read_wav = read(read_path)
    #     sampling_rate_sing, sing_wav = read(sing_path)
    #     stretching_factor = len(read_wav)/len(sing_wav)
    #     from audiotsm import phasevocoder
    #     from audiotsm.io.wav import WavReader, WavWriter
    #     with WavReader(read_path) as reader:
    #         with WavWriter('tmp.tmp', reader.channels, reader.samplerate) as writer:
    #             tsm = phasevocoder(reader.channels, speed=stretching_factor, frame_length=2048)
    #             tsm.run(reader, writer)
    #     stretched_sample_rate_read, stretched_audio_read = read('tmp.tmp')
    #     stretched_sample_rate_sing, stretched_audio_sing = read(sing_path)
    #     print(f'stretching_factor is {stretching_factor}')
    #     print(f'read_wav is {len(read_wav)}')
    #     print(f'sing_wav is {len(sing_wav)}')
    #     print(f'stretched_audio_read is {len(stretched_audio_read)}')
    #     print(f'stretched_audio_sing is {len(stretched_audio_sing)}')
    #
    # args = sys.argv
    # test(args[1], args[2])
def resample():

    args = sys.argv
    path = str(os.path.abspath(args[1]))
    if os.path.isfile(path):
        file = path
        wav, sampling_rate = sf.read(file)
        print("sr: {}".format(sampling_rate))
        scale_factor =  16000 / 22050
        resampled_audio = resize_right.resize(torch.Tensor(wav), out_shape=(len(wav) * scale_factor,))
        sf.write("22050.wav", resampled_audio, 22050)
    else:
        if not(os.path.exists('NHSS_splitted_16000')):
            os.mkdir('NHSS_splitted_16000')
        os.chdir('./NHSS_splitted_16000')
        for file in os.listdir(path):
            if not file.endswith(".wav"):
                continue
            wav, sampling_rate = sf.read(os.path.join(path,file))
            print("sr: {}".format(sampling_rate))
            scale_factor = 16000 / 22050
            resampled_audio = resize_right.resize(torch.Tensor(wav), out_shape=(len(wav) * scale_factor,))
            sf.write(str(file), resampled_audio, 16000)


if __name__ == '__main__':
    preprocess()
    # resample()