import csv
import torch

file_path = r"./data/mockup/source_sets.csv"

def calculate_set_complexity(data:torch.tensor, label:str):
    # Coefficient of variation
    mean = torch.mean(data + 1)
    std = torch.std(data + 1)
    cv = (std / mean) * 100
    print(f'[{label}] Coefficient of variation: {cv.item()}%')

    # Interquartile range
    iqr = torch.quantile(data, 0.75) - torch.quantile(data, 0.25)
    print(f'[{label}] Interquartile range: {iqr.item()}')

    # Entropy
    hist = torch.histc(data, bins=10, min=float(data.min()), max=float(data.max()))
    pdf = hist / torch.sum(hist)
    entropy = -torch.sum(pdf * torch.log2(pdf + torch.finfo(torch.float32).eps))
    print(f'[{label}] Entropy: {entropy.item()}')

    return [label, cv.item(), iqr.item(), entropy.item()]

csv_data = []
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        csv_data.append(row)

labels = csv_data[0]
data = csv_data[1:]
for i in range(len(data)):
    for j in range(len(data[0])):
        data[i][j] = float(data[i][j])

data = torch.tensor(data)
data = data.transpose(0, 1)

result_list = [["Name", "Coefficient of variation", "Interquartile range", "Entropy"]]
for i in range(len(labels)):
    result_list.append(calculate_set_complexity(data=data[i], label=labels[i]))

with open("set_complexity_result.csv", 'w', newline="") as file:
    writer = csv.writer(file)
    for data in result_list:
        writer.writerow(data)