import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import Dataset
import numpy as np
from utils.logger import LoggerHelper



class EEGDataset(Dataset):
    def __init__(self, x_path):
        self.logger=LoggerHelper.get_logger()

        try:
            self.X=np.load(x_path)
            self.X=torch.from_numpy(self.X).float()
            self.logger.info(f"Dataset is loaded: {x_path}, shape: {self.X.shape}")
        except Exception as e:
            self.logger.error(f"Loading data failed with error: {e}")
            raise e
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        signal=self.X[index]
        signal=signal.permute(1,0)
        return signal