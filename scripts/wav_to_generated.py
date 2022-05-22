import os
import argparse
from scipy.io.wavfile import read, write

import sine_pitch_exp
import preprocess

# Global configurations
target_resample_sr = 16000  # sample rate of pre-sample process
resize_method = "resample"  # method for equalization of speech and singing lengths (resample/padding)


class ResampleArgs:
    def __init__(self, data_path, target_resample_sr, text_path, out_dir):
        self.path = data_path
        self.target_sr = target_resample_sr
        self.output_dir = out_dir
        self.text_path = text_path


class StretchArgs:
    def __init__(self, data_path, text_path, out_dir, method):
        self.path = data_path
        self.text_path = text_path
        self.output_dir = out_dir
        self.method = method


class PitchFeaturesArgs:
    def __init__(self, data_path, text_path, out_dir):
        self.path = data_path
        self.text_path = text_path
        self.output_dir = out_dir


def create_txt_file_from_str(wav_path, txt_string, out_dir):
    out_file = out_dir + "/preprocessed_wav_files.txt"
    with open(out_file, "a") as out_f:
        for wav_file in os.listdir(wav_path):
            if (txt_string in wav_file) and ("read" in wav_file):
                out_f.write(wav_file + "\n")

    return out_file


def pre_resample(wav_path, txt_file, output_dir):
    resample_args = ResampleArgs(wav_path, target_resample_sr, txt_file, output_dir)

    resampled_dir = preprocess.resample(resample_args)

    return resampled_dir


def main(args):
    # pre-resample the data
    resampled_dir = pre_resample(args.wav_path, args.txt_file_path, args.output_dir)

    # stretch and resample/padding read to singing max length
    stretch_args = StretchArgs(resampled_dir, args.txt_file_path, args.output_dir, resize_method)
    stretched_dir = preprocess.speech_to_sing_resize(stretch_args, max_len=0)

    # sine pitch extraction
    sine_pitch_exp.main(stretched_dir, args.txt_file_path, args.output_dir)

    # # pitch features extraction
    # pitch_features_args = PitchFeaturesArgs(stretched_dir, args.txt_file_path, args.output_dir)
    # preprocess.pitch_feature_extraction(pitch_features_args)

    if args.inference:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generation of full preprocessed data from speech .wav file or directory of .wav files',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-wav', '--wav_path', metavar='WAV_PATH', type=str, required=True,
                        help='path to the reading audio files directory or to .wav file')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-txt_file', '--txt_file_path', metavar='TXT_FILE_PATH', type=str, default=None,
                       help='path to the file contains the list of speech files names (for directory option)')
    group.add_argument('-txt_str', '--txt_string', metavar='TXT_STRING', type=str, default=None,
                       help='process all the files in the wav directories which contain this string')

    parser.add_argument('-out', "--output_dir", default=os.getcwd(),
                        help='the output directory for saving the resampled file (default: current directory)')

    parser.add_argument('-infer', "--inference", metavar='INFERENCE', type=str, default=None,
                        help='inference of the input after preprocess')

    args = parser.parse_args()
    if not os.path.isdir(args.wav_path):
        print("\nArguments error: wav_path is not a directory\n")
        exit()

    args.wav_path = str(os.path.abspath(args.wav_path))

    # create txt file, in the output directory, of all the file names that contain txt_string (from wav_path directory)
    if args.txt_string is not None:
        args.txt_file_path = create_txt_file_from_str(args.wav_path, args.txt_string, args.output_dir)

    main(args)