import csv
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from recurrent_models import StackedQLSTM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Program started")

# Hyper parameters
input_size = 4          # Quaternion
sequence_length = 100   # Frames
num_layers = 2

hidden_size = 128
num_classes = 4
num_epochs = 3
batch_size = 10
learning_rate = 0.001

# ===>>> Classes <<<===
# Rotation dataset
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
        print(f"Reading: {file_path}")
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
        return torch.tensor(final_data, dtype=torch.float32)
    
    def _prepare_labels_dataset(self, data):
        final_data = []
        for i in range(int(len(data) / input_size)):
            sequence = []
            sequence.append([data[i][0], data[i+1][0], data[i+2][0], data[i+3][0]])
            final_data.append(sequence)
        
        #return final_data
        return torch.tensor(final_data, dtype=torch.float32)


# ===>>> Temporary testing functions <<<===
def predict():
    file_path = r"./data/mockup/training_data (Test).csv"
    labels_path = r"./data/mockup/labels_data (Test).csv"
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

        data = [x[1:] for x in data]
        data = [[float(y) for y in x] for x in data]

    final_data = []
    for i in range(int(len(data) / input_size)):
        sequence = []
        for j in range(sequence_length):
            sequence.append([data[i][j], data[i+1][j], data[i+2][j], data[i+3][j]])
        final_data.append(sequence)

    labels = []
    with open(labels_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row)

        labels = [x[1:] for x in labels]
        labels = [[float(y) for y in x] for x in labels]

    final_labels = []
    for i in range(int(len(labels) / input_size)):
        sequence = []
        sequence.append([labels[i][0], labels[i+1][0], labels[i+2][0], labels[i+3][0]])
        final_labels.append(sequence)

    with torch.no_grad():
        data = torch.tensor(final_data, dtype=torch.float32).to(device)
        l_tensor = torch.tensor(final_labels, dtype=torch.float32).to(device)
        l_tensor = l_tensor.reshape(l_tensor.shape[0], 4)

        output = model(data)
        output = output[:, -1, :] 
        output_list = output.tolist()
        # Zaokrąglenie wartości do 5 miejsc po przecinku
        rounded_list = [[round(x, 5) for x in q] for q in output_list]
        loss = criterion(output, l_tensor)
        print(f"result: {rounded_list}, expected: {l_tensor}, loss: {loss.item():.7f}")



# 1. Creating dataset
print("1. Creating dataset")
training_path = r"./data/mockup/training_data (Medium).csv"
labels_path = r"./data/mockup/labels_data (Medium).csv"
dataset = RotationDataset(training_path, labels_path)

# 2. Splitting dataset
print("2. Splitting dataset")
training_size = int(0.8 * len(dataset))
test_size = len(dataset) - training_size
training_dataset, test_dataset = random_split(dataset, [training_size, test_size])

# 3. Generating DataLoaders
print("3. Generating DataLoaders")
train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# 4. Creating model
print("4. Creating model")
#model = QLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model = StackedQLSTM(input_size, hidden_size, True, num_layers, True).to(device)


class QALLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def quaternion_conjugate(self, q):
        w, v = q[:, 0], q[:, 1:]
        return torch.cat((w.unsqueeze(-1), -v), dim=1)

    def quaternion_multiply(self, q1, q2):
        w1, v1 = q1[:, 0], q1[:, 1:]
        w2, v2 = q2[:, 0], q2[:, 1:]

        w = w1 * w2 - (v1 * v2).sum(dim=1)
        v = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + torch.cross(v1, v2)

        return torch.cat((w.unsqueeze(-1), v), dim=1)

    def forward(self, output: torch.tensor, expected: torch.tensor) -> torch.tensor:
        distance = self.quaternion_multiply(self.quaternion_conjugate(output), expected)
        w = distance[:, 0]
        angles_rad = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
        angles_rad = angles_rad**2
        return torch.mean(angles_rad)


# 5. Loss and optimizer
print("5. Creating criterion and optimizer")
criterion = QALLoss()
criterion_test = QALLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
predict()


# 6. Training loop
print("6. Starting training loop")
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (rotations, labels) in enumerate(train_loader):
        rotations = rotations.to(device)
        labels = labels.to(device)
        labels = labels.reshape(labels.shape[0], 4)

        # Forward
        outputs = model(rotations)
        outputs = outputs[:, -1, :]
        loss = criterion(outputs, labels)

        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss {loss.item()}')

# 7. Test and evaluation
print("7. Starting evaluation")
with torch.no_grad():
    test_loss = []
    n_samples = 0

    for (rotations, labels) in test_loader:
        rotations = rotations.to(device)
        labels = labels.to(device)
        labels = labels.reshape(labels.shape[0], input_size)
        output = model(rotations)
        output = output[:, -1, :] 

        test_loss.append(criterion_test(output, labels).item())

    test_loss = np.array(test_loss)
    loss_mean = np.mean(test_loss)
    loss_std = np.std(test_loss)
    print(f'Loss mean: {loss_mean:.7f}, loss std: {loss_std:.7f}')

predict()