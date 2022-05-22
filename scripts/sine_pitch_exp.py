import crepe
import torch
import numpy as np
import soundfile as sf
import resize_right
import os
import argparse


device = torch.device(0)


def sine_pitch_extract(file_path):
    wav_file, sr = sf.read(file_path)
    wav_file[wav_file <= 0.0001] = 0
    time, pitch, conf, _ = crepe.predict(wav_file, sr, verbose=0, viterbi=True)
    resampled_pitch = resize_right.resize(torch.Tensor(pitch), out_shape=(len(wav_file),))
    sine_pitch = np.sin(2*np.pi*np.cumsum(resampled_pitch)/sr)

    return sine_pitch, sr


def extract_and_save_sine_pitch(wav_path, results_dir, in_type="dir"):
    wav_path = wav_path.strip()  # removing redundant backspaces
    file_name = os.path.basename(os.path.normpath(wav_path))  # the basename of the file
    if in_type == "file":
        sine_pitch, sr = sine_pitch_extract(wav_path)
        out_file_name = file_name.replace('sing', 'sine_pitch')
    else:
        sing_file_name = file_name.replace('read', 'sing')  # the singing file name
        file_path = os.path.dirname(wav_path) + "/" + sing_file_name  # the singing path
        sine_pitch, sr = sine_pitch_extract(file_path)
        out_file_name = file_name.replace('read', 'sine_pitch')

    out_file_path = results_dir + "/" + out_file_name
    sf.write(out_file_path, sine_pitch, sr)


def main(wav_path, txt_path, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    results_dir = output_dir + '/sine_pitch_results/'
    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    if os.path.isfile(wav_path):
        extract_and_save_sine_pitch(wav_path, results_dir, in_type="file")
    else:
        with open(txt_path, "r") as names_file:
            for line in names_file:
                file_path = wav_path+line
                extract_and_save_sine_pitch(file_path, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracting sine-pitch from .wav audio waveform',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('wav_path', metavar='wav_path', type=str,
                        help='path to the audio files directory or to .wav file')
    parser.add_argument('-txt', '--txt_path', metavar='txt_path', type=str, default=None,
                       help='path to the file contains the list of speech files names')

    parser.add_argument('-out', "--output_dir", default=os.getcwd(),
                                 help='the output directory for saving the resampled file (default: current directory)')

    args = parser.parse_args()

    if os.path.isdir(args.wav_path) and (args.txt_path is None):
        print("\nArguments error: a directory path was received without a txt file\n")
        exit()

    if not os.path.isdir(args.wav_path) and not os.path.isfile(args.wav_path):
        print("\nArguments error: the wav_path is not to a valid file\n")
        exit()


    main(args.wav_path, args.txt_path, args.output_dir)