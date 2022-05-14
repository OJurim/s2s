import numpy as np
import torch

a = np.zeros([1, 5, 6])
b = np.ones([1, 5, 6])
c = np.zeros([1, 10, 6])

for i in range(5):
    c[:, 2*i+1, :] = b[:, i, :]
    c[:, 2 * i, :] = a[:, i, :]

print((c.shape))