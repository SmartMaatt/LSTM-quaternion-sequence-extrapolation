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

def generate_model_file_name(model_name, loss_name, set_name, num_epochs):
       return f"{model_name}_{loss_name}_{set_name}_epochs{num_epochs}"

class ModelType(Enum):
        LSTM = 0
        QLSTM = 1
        VectorizedQLSTM = 2