import os
import argparse
from scipy.io.wavfile import read, write
import soundfile as sf
import torch
import numpy as np
import resize_right
from f0_package import crepe_pytorch, spectral_feats
from librosa.util import normalize


def audio_stretcher(full_path_read, full_path_sing, stretching_factor=0.5, segment_size=3500000,  method="resample"):

    # stretching by stretching_size factor
    from audiotsm import phasevocoder
    from audiotsm.io.wav import WavReader, WavWriter
    with WavReader(full_path_read) as reader:
        with WavWriter('tmp.tmp', reader.channels, reader.samplerate) as writer:
            tsm = phasevocoder(reader.channels, speed=stretching_factor)
            tsm.run(reader, writer)

    stretched_audio_read, stretched_sample_rate_read = sf.read('tmp.tmp')
    os.remove("tmp.tmp")
    stretched_audio_sing, stretched_sample_rate_sing = sf.read(full_path_sing)

    if method == "padding":
        # padding with zeros to segment_size length
        padding_size_read = segment_size - len(stretched_audio_read)
        stretched_audio_read = np.pad(stretched_audio_read, (0, padding_size_read), 'constant', constant_values=0)
        padding_size_sing = segment_size - len(stretched_audio_sing)
        stretched_audio_sing = np.pad(stretched_audio_sing, (0, padding_size_sing), 'constant', constant_values=0)
        dec_factor = 1
    elif method == "resample":
        long_wav = max(len(stretched_audio_read), len(stretched_audio_sing))
        short_wav = min(len(stretched_audio_read), len(stretched_audio_sing))
        padding_size = long_wav - short_wav
        if len(stretched_audio_read) > len(stretched_audio_sing):
            stretched_audio_sing = np.pad(stretched_audio_sing, (0, padding_size), 'constant', constant_values=0)
        else:
            stretched_audio_read = np.pad(stretched_audio_read, (0, padding_size), 'constant', constant_values=0)
        # referring segment size as maximal length
        samples_num = len(stretched_audio_read)
        dec_factor = segment_size/samples_num
        stretched_audio_sing = resize_right.resize(torch.Tensor(stretched_audio_sing), out_shape=(segment_size,))
        stretched_audio_read = resize_right.resize(torch.Tensor(stretched_audio_read), scale_factors=None, out_shape=(segment_size,))

    read_fs = int(stretched_sample_rate_read*dec_factor)
    sing_fs = int(stretched_sample_rate_sing*dec_factor)

    return stretched_audio_read, read_fs, stretched_audio_sing, sing_fs


def speech_to_sing_resize(args, max_len=0):
    text_path = str(os.path.abspath(args.text_path))
    wav_path = str(os.path.abspath(args.path))
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.getcwd()
    results_dir = out_dir + '/%s_speech_to_sing_resized_%s/' % (text_path.split('/')[-1].split('.')[0], args.method)
    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    if max_len == 0:
        with open(text_path, 'r') as file_list:
            while True:
                line_read = file_list.readline()
                if not line_read: break
                line_sing = line_read.replace("read", "sing")
                full_path_sing = wav_path + "/" + line_sing[:-1]
                sing_wav, sampling_rate_sing = sf.read(full_path_sing)
                if len(sing_wav) > max_len:
                    max_len = len(sing_wav)

    print(f'max_len is {max_len}')

    with open(text_path, 'r') as file_list:
        while True:
            line_read = file_list.readline()
            if not line_read:
                break
            line_sing = line_read.replace("read", "sing")
            full_path_read = wav_path + "/" + line_read[:-1]
            sampling_rate_read, read_wav = read(full_path_read)
            full_path_sing = wav_path + "/" + line_sing[:-1]
            sampling_rate_sing, sing_wav = read(full_path_sing)
            stretching_factor = len(read_wav)/len(sing_wav)
            r_wav, r_fs, s_wav, s_fs = audio_stretcher(full_path_read, full_path_sing, stretching_factor=stretching_factor,
                                                       segment_size=max_len, method=args.method)
            sf.write(results_dir + line_sing[:-1], s_wav, s_fs)
            sf.write(results_dir + line_read[:-1], r_wav, r_fs)

    return results_dir

def resample(args):
    data_path = args.path
    target_sr = int(args.target_sr)
    out_dir = args.output_dir
    text_path = args.text_path

    if out_dir is None:
        out_dir = os.getcwd()
    path = str(os.path.abspath(data_path))
    if os.path.isfile(path):
        file = path
        wav, sampling_rate = sf.read(file)
        print("input file sampling rate: {}".format(sampling_rate))
        scale_factor = target_sr / sampling_rate
        resampled_audio = resize_right.resize(torch.Tensor(wav), out_shape=(len(wav) * scale_factor,))
        file_name, suffix = file.split('/')[-1].split('.')
        out_path = out_dir + '/%s_%s.%s' % (file_name, target_sr, suffix)
        sf.write(out_path, resampled_audio, target_sr)

        return out_dir
    else:
        results_dir = out_dir + '/%s_resampled_%s/' % (path.split('/')[-1], target_sr)
        if not(os.path.exists(results_dir)):
            os.makedirs(results_dir)

        def resample_one_file(path, file):
            wav, sampling_rate = sf.read(os.path.join(path, file))
            print("{} sampling rate: {}".format(str(file), sampling_rate))
            scale_factor = target_sr / sampling_rate
            resampled_audio = resize_right.resize(torch.Tensor(wav), out_shape=(len(wav) * scale_factor,))
            out_path = results_dir + str(file)
            sf.write(out_path, resampled_audio, target_sr)

        if text_path is None:
            for file in os.listdir(path):
                if not file.endswith(".wav"):
                    continue
                resample_one_file(path, file)
        else:
            with open(text_path, 'r') as file_list:
                while True:
                    line_read = file_list.readline()
                    if not line_read:
                        break
                    line_sing = line_read.replace("read", "sing")
                    resample_one_file(path, line_read[:-1])
                    resample_one_file(path, line_sing[:-1])

        return results_dir


def pitch_feature_extraction(args):
    MAX_WAV_VALUE = 32768.0
    text_path = args.text_path

    device = torch.device('cuda:{:d}'.format(0))
    original_dirpath = '/home/ohadmochly@staff.technion.ac.il/git_repo'  # should run from git_repo
    small_crepe = crepe_pytorch.load_crepe(os.path.join(original_dirpath, 'scripts/f0_package/small.pth'), device, 'small')

    wav_path = str(os.path.abspath(args.path))
    wav_folder_name = wav_path.split('/')[-1]
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.getcwd()
    results_dir = out_dir + '/%s_pitch_features_extraction_results/' % str(wav_folder_name)
    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    def extract_features_one_file(wav_path, wavname, results_dir):
        audio, sampling_rate = sf.read(os.path.join(wav_path, wavname))
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = torch.autograd.Variable(audio.to(device, non_blocking=True))
        audio = audio.unsqueeze(1)

        in_features = spectral_feats.py_get_activation(audio.squeeze(1), sampling_rate, small_crepe,
                                                       layer=18, grad=False, sampler=None)
        in_features = in_features.detach()

        pitchname = wavname.replace("sing", "pitch_feats")
        result_file_path = results_dir + '/' + pitchname.split('.')[0] + '.pt'
        torch.save(in_features, result_file_path)

    if text_path is None:
        for wavname in os.listdir(wav_path):
            if "sing" not in wavname:
                continue
            extract_features_one_file(wav_path, wavname, results_dir)
    else:
        with open(text_path, 'r') as file_list:
            while True:
                line_read = file_list.readline()
                if not line_read:
                    break
                line_sing = line_read.replace("read", "sing")[:-1]
                extract_features_one_file(wav_path, line_sing, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing audio data options:\n\t'
                                                 '1. Resampling of single file or directory of .wav files\n\t'
                                                 '2. Speech to sing resize - stretching speech to sing size and then resample/pad to max sing size\n\t'
                                                 '3. Pitch features extraction by CREPE network',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('path', metavar='path', type=str,
                        help='path to the directory or to .wav file (only for resample option)')

    subparsers = parser.add_subparsers(dest='action', help='desired action to perform')
    subparsers.required = True

    resample_parser = subparsers.add_parser('resample')
    resample_parser.set_defaults(func=resample)
    resample_parser.add_argument('target_sr', help='the target sampling rate after resampling')
    resample_parser.add_argument("-txt", "--text_path", default=None, help='the text file that contains the .wav file names to be resampled')
    resample_parser.add_argument('-out', "--output_dir", default=None, help='the output directory for saving the resampled file')

    preprocess_parser = subparsers.add_parser('resize')
    preprocess_parser.set_defaults(func=speech_to_sing_resize)
    preprocess_parser.add_argument("text_path", help='the text file that contains the .wav file names to be preprocessed')
    preprocess_parser.add_argument('-out', "--output_dir", default=None, help='the output directory for saving the preprocessed data')
    preprocess_parser.add_argument('-method', choices=['resample', 'padding'], default='resample', help='the method for equalization of speech and singing lengths')

    f0_parser = subparsers.add_parser('pitch_features')
    f0_parser.set_defaults(func=pitch_feature_extraction)
    f0_parser.add_argument("-txt", "--text_path", default=None, help='the text file that contains the .wav file names for extraction')
    f0_parser.add_argument('-out', "--output_dir", default=None, help='the output directory for saving the pitch features data')

    args = parser.parse_args()


    args.func(args)