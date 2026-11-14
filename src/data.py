import numpy as np
import torch
from torch.utils.data import Dataset


class TokenWindows(Dataset):
    def __init__(self, path, window, max_tokens=-1):
        self.data = np.load(path, mmap_mode="r")
        total = len(self.data) if max_tokens <= 0 else min(len(self.data), max_tokens)
        self.window = window
        self.num_windows = total // window
        if self.num_windows == 0:
            raise ValueError(f"{path} has fewer than {window} tokens")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.window
        return torch.from_numpy(self.data[start:start + self.window].astype(np.int64))
