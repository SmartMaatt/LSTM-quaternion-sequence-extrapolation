import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

import recurrent_models as rm
from dataset_initializer import RotationDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Program started")

# Hyper parameters
input_size = 4          # Quaternion
sequence_length = 100   # Frames
batch_size = 10

show_evaluation = True
model_path = rf"./models/lstm_mse_batch10_epochs3.pth"


# 1. Creating dataset
print("1. Creating dataset")
# training_path = r"./data/mockup/training_data (Medium).csv"
# labels_path = r"./data/mockup/labels_data (Medium).csv"
training_path = r"./data/mockup/large/training_data.csv"
labels_path = r"./data/mockup/large/labels_data.csv"
dataset = RotationDataset(training_path, labels_path, input_size, sequence_length)


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
print("4. Reading model")
try:
    model = torch.load(model_path).to(device)
except FileNotFoundError:
    print(f"Model file {model_path} doesn't exist")
    sys.exit()
print(f"Model type: {type(model)}")


# 5. Criterion
print("5. Creating criterion")
print("Evaluation criterion: QALLoss function")
criterion_eval = rm.QALLoss()


# 6. Test and evaluation
print("6. Starting evaluation")
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