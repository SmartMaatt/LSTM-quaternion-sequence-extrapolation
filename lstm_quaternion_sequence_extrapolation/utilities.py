import torch
from enum import Enum

def normalize_quaternions(q, dim):
        norms = torch.norm(q, dim=dim, keepdim=True)
        return q / norms

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds_left = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {seconds_left:.2f}s"

class ModelType(Enum):
        LSTM = 0
        QLSTM = 1
        VectorizedQLSTM = 2