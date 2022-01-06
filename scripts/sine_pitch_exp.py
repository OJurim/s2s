import crepe
import torch
import numpy as np
import soundfile as sf
import resize_right

wav_file, sr = sf.read("./20_for_inference/sing/F02_S05_100_sing.wav")

device = torch.device(0)
# cr = crepe.predict("full").to(device)
time, pitch, _, _ = crepe.predict(wav_file, sr)
sine_pitch = resize_right.resize(torch.Tensor(pitch), out_shape=(len(wav_file),))
sine_pitch = np.sin(2*np.pi*np.cumsum(sine_pitch)/sr)
sf.write("./old_test_files/sine_of_a_pitch.wav", sine_pitch, sr)
