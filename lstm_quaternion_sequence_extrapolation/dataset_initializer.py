import csv
import torch
from torch.utils.data import Dataset

class RotationDataset(Dataset):
    def __init__(self, training_path, labels_path, input_size, sequence_length):
        super(RotationDataset, self).__init__()
        self.training_path = training_path
        self.labels_path = labels_path
        self.input_size = input_size
        self.sequence_length = sequence_length

        self.training_data = self._read_dataset(training_path)
        self.labels_data = self._read_dataset(labels_path)

        self.training_data = self._prepare_training_dataset(self.training_data, input_size, sequence_length)
        self.labels_data = self._prepare_labels_dataset(self.labels_data, input_size)
        self.n_samples = self.training_data.size()[0]
    
    def __getitem__(self, index):
        return self.training_data[index], self.labels_data[index]
    
    def __len__(self):
        return self.n_samples

    def _read_dataset(self, file_path: str):
        print(f"Reading: {file_path}")
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)

        data = [x[1:] for x in data]
        data = [[float(y) for y in x] for x in data]
        return data

    def _prepare_training_dataset(self, data, input_size, sequence_length):
        final_data = []
        for i in range(int(len(data) / input_size)):
            sequence = []
            for j in range(sequence_length):
                sequence.append([data[i][j], data[i+1][j], data[i+2][j], data[i+3][j]])
            final_data.append(sequence)
        
        #return final_data
        return torch.tensor(final_data)
    
    def _prepare_labels_dataset(self, data, input_size):
        final_data = []
        for i in range(int(len(data) / input_size)):
            sequence = []
            sequence.append([data[i][0], data[i+1][0], data[i+2][0], data[i+3][0]])
            final_data.append(sequence)
        
        #return final_data
        return torch.tensor(final_data)