import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 4          # Quaternion
sequence_length = 100   # Frames
num_layers = 2

hidden_size = 128
num_classes = 4
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Rotation data
class RotationDataset(Dataset):
    def __init__(self, training_path, labels_path):
        super(RotationDataset, self).__init__()
        self.training_path = training_path
        self.labels_path = labels_path

        self.training_data = self._read_dataset(training_path)
        self.labels_data = self._read_dataset(labels_path)

        self.training_data = self._prepare_training_dataset(self.training_data)
        self.labels_data = self._prepare_labels_dataset(self.labels_data)
        self.n_samples = self.training_data.size()[0]
    
    def __getitem__(self, index):
        return self.training_data[index], self.labels_data[index]
    
    def __len__(self):
        return self.n_samples

    def _read_dataset(self, file_path: str):
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)

        data = [x[1:] for x in data]
        data = [[float(y) for y in x] for x in data]
        return data

    def _prepare_training_dataset(self, data):
        final_data = []
        for i in range(int(len(data) / input_size)):
            sequence = []
            for j in range(sequence_length):
                sequence.append([data[i][j], data[i+1][j], data[i+2][j], data[i+3][j]])
            final_data.append(sequence)
        
        #return final_data
        return torch.tensor(final_data)
    
    def _prepare_labels_dataset(self, data):
        final_data = []
        for i in range(int(len(data) / input_size)):
            sequence = []
            sequence.append([data[i][0], data[i+1][0], data[i+2][0], data[i+3][0]])
            final_data.append(sequence)
        
        #return final_data
        return torch.tensor(final_data)

training_path = r"./data/mockup/training_data_test.csv"
labels_path = r"./data/mockup/labels_data_test.csv"

dataset = RotationDataset(training_path, labels_path)

training_size = int(0.8 * len(dataset))
test_size = len(dataset) - training_size
training_dataset, test_dataset = random_split(dataset, [training_size, test_size])

train_loader = torch.utils.data.DataLoader(
    dataset=training_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first -> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))     # out: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]                 # out: [Sentiment classification!]
        out = self.fc(out)
        return out
    
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (rotations, labels) in enumerate(train_loader):
        rotations = rotations.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(rotations)
        loss = criterion(outputs, labels)

        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss {loss.item():.4f}')


# Test and evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for (rotations, lables) in test_loader:
        rotations = rotations.to(device)
        lables = labels.to(device)
        output = model(rotations)

        # [value, index]
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy {acc}')