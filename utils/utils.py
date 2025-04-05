import os
import random
import sys
import numpy as np
import torch

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)  # make experiments reproducible but slow
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # make experiments reproducible but slow
    print(f"Random seed set as {seed}")


def home_folder():
    home = os.environ['HOME']
    return home