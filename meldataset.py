import math
import os
import random
import torch
import torchvision as tv
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                          # + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0])
                            # + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files, a.input_pitch_dir


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, pitch_path, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None, stretching=False):


        # added 31.5.21 to prevent empty data loading

        #
        self.audio_files = training_files
        random.seed(2345)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav_read = None
        self.cached_wav_sing = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.stretching = stretching
        self.pitch_path = pitch_path

    def __getitem__(self, index):

        # if (index % 2) == 1:
            # return torch.FloatTensor([1]*80), torch.FloatTensor([1]*80), torch.FloatTensor([]), torch.FloatTensor([1]*80)
        #print("the index: ", index)
        #if index == len(self.audio_files)-1:
         #   print(index, len(self.audio_files)-1)
        #    return #torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([]), torch.FloatTensor([])
        filename_read = self.audio_files[index]
        filename_sing = filename_read.replace("read", "sing")
        filename_pitch = filename_read.replace("read", "pitch").split('/')[-1].split('.')[0]
        filename_pitch = filename_pitch + '.pt'
        # print((filename_read))
        # print(type(filename_sing))
        # 10.9.21
        sampling_rate_read = -1
        sampling_rate_sing = -1
        #
        if self._cache_ref_count == 0:
            audio_read, sampling_rate_read = load_wav(filename_read)
            audio_sing, sampling_rate_sing = load_wav(filename_sing)
            # resample to 16000
            # audio_read = librosa.resample(audio_read, sampling_rate_read, 16000)
            # audio_sing = librosa.resample(audio_sing, sampling_rate_sing, 16000)

            audio_read = audio_read / MAX_WAV_VALUE
            audio_sing = audio_sing / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio_read = normalize(audio_read) * 0.95
                audio_sing = normalize(audio_sing) * 0.95
            self.cached_wav_read = audio_read
            self.cached_wav_sing = audio_sing
            # if sampling_rate_read != self.sampling_rate:
            #     raise ValueError("{} read SR doesn't match target {} SR".format(
            #         sampling_rate_read, self.sampling_rate))
            # if sampling_rate_sing != self.sampling_rate:
            #     raise ValueError("{} sing SR doesn't match target {} SR".format(
            #         sampling_rate_sing, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio_read = self.cached_wav_read
            audio_sing = self.cached_wav_sing
            self._cache_ref_count -= 1

        audio_read = torch.FloatTensor(audio_read)
        audio_read = audio_read.unsqueeze(0)
        audio_sing = torch.FloatTensor(audio_sing)
        audio_sing = audio_sing.unsqueeze(0)
        # print(audio_read.shape)
        # print(audio_sing.shape)
        # init split = false for now
        if not self.fine_tuning:
            # if self.split:
            #     if audio_read.size(1) >= self.segment_size:
            #         max_audio_start = audio_read.size(1) - self.segment_size
            #         audio_start = random.randint(0, max_audio_start)
            #         audio_read = audio_read[:, audio_start:audio_start+self.segment_size]
            #         audio_sing = audio_sing[:, audio_start:audio_start + self.segment_size]
            #     else:
            #         audio_read = torch.nn.functional.pad(audio_read, (0, self.segment_size - audio.size(1)), 'constant')
            #         audio_sing = torch.nn.functional.pad(audio_sing, (0, self.segment_size - audio.size(1)), 'constant')

            mel_read = mel_spectrogram(audio_read, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            # if self.split:
            #     frames_per_seg = math.ceil(self.segment_size / self.hop_size)
            #
            #     if audio.size(1) >= self.segment_size:
            #         mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
            #         mel = mel[:, :, mel_start:mel_start + frames_per_seg]
            #         audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            #     else:
            #         mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
            #         audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_sing_loss = mel_spectrogram(audio_sing, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        # added 31.5.21 to prevent empty data loading
        #return torch.tensor(mel_read.squeeze()), torch.tensor(audio_sing.squeeze(0)), filename_read, mel_sing_loss.squeeze()
        #print(len(mel_read.squeeze()))
        if mel_read is None or audio_sing is None or filename_read is None or mel_sing_loss is None:
            print(filename_read)
        # 10.9.21

        pitch_file_path = self.pitch_path + '/' + filename_pitch

        return mel_read.squeeze(), audio_sing.squeeze(0), filename_read, \
               mel_sing_loss.squeeze(), sampling_rate_read, sampling_rate_sing, pitch_file_path
        #
    def __len__(self):
        return len(self.audio_files)


