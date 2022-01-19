import crepe
import torch
import numpy as np
import soundfile as sf
import resize_right
import os

device = torch.device(0)

if not os.path.isdir("/home/ohadmochly@staff.technion.ac.il/git_repo/small_data_files/sine_pitch"):
    os.makedirs("/home/ohadmochly@staff.technion.ac.il/git_repo/small_data_files/sine_pitch")

with open ("/home/ohadmochly@staff.technion.ac.il/git_repo/small_data_files/train_NHSS_small.txt", "r") as names_file:
    for line in names_file:
        line = line.strip()
        file_name = line.replace('read', 'sing')
        file_name = "/home/ohadmochly@staff.technion.ac.il/git_repo/20_for_inference/sing/"+file_name
        wav_file, sr = sf.read(file_name)

        time, pitch, _, _ = crepe.predict(wav_file, sr)
        sine_pitch = resize_right.resize(torch.Tensor(pitch), out_shape=(len(wav_file),))
        sine_pitch = np.sin(2*np.pi*np.cumsum(sine_pitch)/sr)
        out_file_name = line.replace('read', 'sine_pitch')
        out_file_name = "/home/ohadmochly@staff.technion.ac.il/git_repo/small_data_files/sine_pitch/"+out_file_name
        sf.write(out_file_name, sine_pitch, sr)

