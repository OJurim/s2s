from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

import librosa
import soundfile as sf

#for file_name in os.listdir(FOLDER_PATH):
#    with wave.open(file_name, "rb") as wave_file:
#        frame_rate = wave_file.getframerate()

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax, h.upsample_rates)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            pitch_filename = filename.replace('read', 'sine_pitch')
            pitch, _ = load_wav(os.path.join(a.input_pitch_dir, pitch_filename))
            # wav, sr = sf.read(os.path.join(a.input_wavs_dir, filname))
            # full_file_name = os.path.join(a.input_wavs_dir, filname)
            #wav_44100, sr = librosa.load(full_file_name, sr=44100)

            print('The sample rate is: {0}'.format(sr))#DEBUG
            #wav = librosa.resample(wav, sr, 22050)
            #write('out'+filname, 22050, wav)#DEBUG
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            pitch = pitch / MAX_WAV_VALUE
            pitch = torch.FloatTensor(pitch).to(device)

            upsample_factor = np.prod(h.upsample_rates, initial=1)
            if (len(pitch.squeeze(0)) % upsample_factor) != 0:
                padding_size = (0, upsample_factor - (len(pitch.squeeze(0)) % upsample_factor))
                pitch = torch.nn.functional.pad(pitch, padding_size, mode='constant', value=0).unsqueeze(0)

            # pitch = get_mel(pitch.unsqueeze(0))
            y_g_hat = generator(x, pitch.unsqueeze(0))
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, sr, audio)
            # sf.write(output_file, audio, sr)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--input_pitch_dir', default='/home/ohadmochly@staff.technion.ac.il/git_repo/small_data_files/sine_pitch/')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    # config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'generated/archive/config.json')
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config_v1_f0.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(h.seed)
    #     device = torch.device('cuda')
    # else:
    device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

