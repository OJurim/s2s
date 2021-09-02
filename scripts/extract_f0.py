import torchcrepe
import datetime
import argparse
import os
import sys
import numpy


def save_pitch(pitch, out_file_name):
    pitch_np = pitch.squeeze(0).numpy()[:]
    pitch_str = numpy.array2string(pitch_np, max_line_width=numpy.inf)
    with open(out_file_name, 'w+') as out:
        out.write(pitch_str)


def estimate_pitch_file(file_path):
    audio, sr = torchcrepe.load.audio(file_path)

    # Here we'll use a 5 millisecond hop length
    hop_length = int(sr / 200.)

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 550

    # Select a model capacity--one of "tiny" or "full"
    model = 'tiny'

    # Choose a device to use for inference
    # device = 'cuda:0'

    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048

    tic = datetime.datetime.now()

    # Compute pitch using first gpu
    pitch = torchcrepe.predict(audio,
                               sr,
                               hop_length,
                               fmin,
                               fmax,
                               model,
                               batch_size=batch_size)

    toc = datetime.datetime.now()
    run_time = toc - tic

    return pitch, run_time.total_seconds()


def estimate_pitch_dir(dir_path):
    total_run_time = 0
    for file in os.listdir(dir_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dir_path, file)
            pitch, run_time = estimate_pitch_file(file_path)
            out_file = file_path.split(".")[-2] + "_pitch.txt"
            save_pitch(pitch, out_file)
            total_run_time = total_run_time + run_time

    return total_run_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate pitch for all .wav audio files in a directory and write it '
                                                 'to a file for each audio file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', metavar='directory_path', type=str, help='path to directory that contains wav files')
    group.add_argument('-f', metavar='file_path', type=str, help='path to .wav file')
    args = parser.parse_args()

    dir_path = args.d
    file_path = args.f
    if dir_path is not None:
        if not os.path.isdir(dir_path):
            print("Argument \"{}\" is not directory path".format(dir_path))
            sys.exit(-1)
        dir_path = os.path.realpath(dir_path)
        total_run_time = estimate_pitch_dir(dir_path)

    else:
        if not os.path.isfile(file_path):
            print("Argument \"{}\" is not file path".format(file_path))
            sys.exit(-1)
        if not file_path.endswith(".wav"):
            print("Argument \"{}\" is not .wav file".format(file_path))
            sys.exit(-1)

        file_path = os.path.realpath(file_path)
        pitch, total_run_time = estimate_pitch_file(file_path)
        out_file = file_path.split(".")[-2] + "_pitch.txt"
        save_pitch(pitch, out_file)

    # Print the overall calculation time
    print("\nTotal run time: " + str(total_run_time) + " sec\n")
