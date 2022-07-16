import os
import argparse
from scipy.io.wavfile import read, write
import soundfile as sf
import torch
import numpy as np
import resize_right
from f0_package import crepe_pytorch, spectral_feats

from scipy.signal import lfilter
from audiolazy import lazy_lpc


def read_wav(wav_file):
    wav_file_path = str(os.path.abspath(wav_file))
    wav, sampling_rate = sf.read(wav_file_path)

    return wav, sampling_rate


def preemp(inp, p):
    """Pre-emphasis filter."""
    return lfilter([1., -p], 1, inp)


def predict(wav, sampling_rate):
    x = wav*np.hamming(len(wav))
    x = preemp(x, -0.63)

    lpc_polynom = lazy_lpc.lpc(x, 8)
    lpc_roots = lpc_polynom.numpoly.roots

    # other way to get the roots
    lpc_coeffs = list(lpc_polynom.numpoly.values())
    np_roots = np.roots(lpc_coeffs)

    lpc_roots_imag = np.imag(lpc_roots) # np.imag(np_roots)

    # Because the LPC coefficients are real-valued, the roots occur in complex conjugate pairs.
    # Retain only the roots with one sign for the imaginary part and determine the angles corresponding to the roots.
    rts = []
    [rts.append(lpc_roots[idx]) for idx, root in enumerate(lpc_roots_imag) if root >= 0]

    # ang = np.arctan2(np.imag(rts), np.real(rts))
    rts_angles = np.angle(rts) * (sampling_rate/(2*np.pi))

    freqs = np.sort(rts_angles)
    indices = np.argsort(rts_angles)

    bw_calc = lambda a: 1/2*(sampling_rate/(2*np.pi))*np.log(abs(a))
    bw = [bw_calc(rts[idx]) for idx in indices]

    formants = []
    for freq, bw in zip(freqs, bw):
        if (freq > 90) and (bw < 400):
            formants.append(freq)

    print("The formants are: " + str(formants))
    return formants


def main(wav_file, begin_time, end_time):
    wav, sr = read_wav(wav_file)
    if begin_time and end_time:
        begin_sample = int(np.floor(sr*begin_time))
        end_sample = int(np.floor(sr*end_time))
        if end_sample > len(wav):
            raise Exception("end time is larger than the wav file length")
        time_ranged_wav = wav[begin_sample:end_sample]
        formants = predict(time_ranged_wav, sr)
    else:
        print(len(wav)*(1/sr))
        formants = predict(wav, sr)

    return formants


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Estimation and tracking of formants.')
    parser.add_argument('wav_file', default='', help="WAV audio filename (single vowel or an whole utternace)")
    parser.add_argument('--begin', '-b', help="beginning time in the WAV file", default=None, type=float)
    parser.add_argument('--end', '-e', help="end time in the WAV file", default=None, type=float)
    args = parser.parse_args()

    main(args.wav_file, args.begin, args.end)
