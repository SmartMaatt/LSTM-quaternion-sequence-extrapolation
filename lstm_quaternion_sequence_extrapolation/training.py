import sys
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

import recurrent_models as rm
from dataset_initializer import RotationDataset
from utilities import seconds_to_hms, ModelType

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

model_type = ModelType.LSTM
is_qal_loss = True

show_evaluation = False
model_dir = rf"./models"


# 1. Creating dataset
print("\n1. Creating dataset")
training_path = r"./data/mockup/training_data (Medium).csv"
labels_path = r"./data/mockup/labels_data (Medium).csv"
# training_path = r"./data/mockup/large/training_data.csv"
# labels_path = r"./data/mockup/large/labels_data.csv"
dataset = RotationDataset(training_path, labels_path, input_size, sequence_length)


# 2. Splitting dataset
print("\n2. Splitting dataset")
training_size = int(0.8 * len(dataset))
test_size = len(dataset) - training_size
training_dataset, test_dataset = random_split(dataset, [training_size, test_size])


# 3. Generating DataLoaders
print("3. Generating DataLoaders")
train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


# 4. Creating model
print("4. Creating model")
if model_type == ModelType.LSTM:
    print(f"Model: LSTM")
    model = rm.LSTM(input_size, hidden_size, num_layers, num_classes, device).to(device)

elif model_type == ModelType.QLSTM:
    print(f"Model: QLSTM")
    model = rm.StackedQLSTM(input_size, hidden_size, num_layers, batch_first=True, device=device).to(device)

elif model_type == ModelType.VectorizedQLSTM:
    print(f"Model: Vectorized QLSTM")
    model = rm.VectorizedStackedQLSTM(input_size, hidden_size, num_layers, batch_first=True, device=device).to(device)

else:
    print("Incorrect model type!")
    sys.exit()

print(f"Sequence length: {sequence_length}")
print(f"Layers: {num_layers}")
print(f"Hidden size: {hidden_size}")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")


# 5. Loss and optimizer
print("\n5. Creating criterion and optimizer")
if is_qal_loss:
    print("Training criterion: QALLoss function")
    criterion = rm.QALLoss()
else:
    print("Training criterion: MSELoss function")
    criterion = nn.MSELoss()

print("Evaluation criterion: QALLoss function")
criterion_eval = nn.MSELoss()

print("Optimizer: Adam")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"Learning rate: {learning_rate}")


# 6. Training loop
print("\n6. Starting training loop")
n_total_steps = len(train_loader)
start_time = time.time()

model.train()
for epoch in range(num_epochs):
    for i, (rotations, labels) in enumerate(train_loader):
        rotations = rotations.to(device)
        labels = labels.to(device)
        labels = labels.reshape(labels.shape[0], 4)

        # Forward
        outputs = model(rotations)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss {loss.item():.7f}, time: {(time.time() - start_time):.2f}s')
print(f"Learning took {seconds_to_hms(time.time() - start_time)}, [{time.time() - start_time}]")


# 7. Test and evaluation
print("\n7. Starting evaluation")
model.eval()
with torch.no_grad():
    test_loss = []
    n_samples = 0

    for (rotations, labels) in test_loader:
        rotations = rotations.to(device)
        labels = labels.to(device)
        labels = labels.reshape(labels.shape[0], input_size)
        output = model(rotations)

        test_loss.append(criterion_eval(output, labels).item())

        # Show evaluation results
        if show_evaluation:
            output = output.tolist()
            labels = labels.tolist()
            n_samples += 1

            for i in range(len(labels)):
                print(f"output: {output[i]}, expected: {labels[i]}")
            print(f"loss: {test_loss[n_samples - 1]}")

    test_loss = np.array(test_loss)
    loss_mean = np.mean(test_loss)
    loss_std = np.std(test_loss)
    print(f'Loss mean: {loss_mean:.7f}, loss std: {loss_std:.7f}')


# 8. Saving model
print("\n8. Saving model")

# Configure file name
model_name = ""
if model_type == ModelType.LSTM:
    model_name = "LSTM"
elif model_type == ModelType.QLSTM:
    model_name = "QLSTM"
elif model_type == ModelType.VectorizedQLSTM:
    model_name = "VectorizedQLSTM"

loss_name = ""
if is_qal_loss:
    loss_name = "qal"
else:
    loss_name = "mse"

model_path = rf"{model_dir}/{model_name}_{loss_name}_batch{batch_size}_epochs{num_epochs}.pth"
print(f"Path: {model_path}")

torch.save(model, model_path)
print("Saving successed")