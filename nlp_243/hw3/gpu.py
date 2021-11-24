import os

import torch
import numpy as np


def get_free_gpu():
    available_mem = []
    for i in range(6):
        t = torch.cuda.get_device_properties(i).total_memory
        r = torch.cuda.memory_reserved(i)
        a = torch.cuda.memory_allocated(i)
        available_mem.append(t - r)

    return int(np.argmax(available_mem))


cmd = "export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6"
os.popen(cmd)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")
if torch.cuda.is_available():
    free_gpu_id = get_free_gpu()
    torch.cuda.set_device(free_gpu_id)
