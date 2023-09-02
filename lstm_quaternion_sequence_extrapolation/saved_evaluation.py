import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

import recurrent_models as rm
from dataset_initializer import RotationDataset
from utilities import *

def saved_evaluation(
        input_size = 4,             # Quaternion
        sequence_length = 100,      # Frames
        num_layers = 2, 

        hidden_size = 128, 
        num_classes = 4, 
        num_epochs = 4, 
        batch_size = 10, 
        trained_on_qal = True,

        model_type = ModelType.LSTM,
        is_qal_loss = False,
        show_evaluation = False,
        calculate_accuracy = True,
        max_acc_round_point = 7,

        model_dir = rf"./models",
        set_name = "hip",
        training_path = r"./data/mockup/large/training_data.csv",
        labels_path = r"./data/mockup/large/labels_data.csv"
):
    # Random seed configuration
    torch.manual_seed(303)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n>>> Saved evaluation procedure started <<<")

    # File name configuration
    model_name = ""
    if model_type == ModelType.LSTM:
        model_name = "LSTM"
    elif model_type == ModelType.QLSTM:
        model_name = "QLSTM"
    elif model_type == ModelType.VectorizedQLSTM:
        model_name = "VectorizedQLSTM"

    loss_name = ""
    if trained_on_qal:
        loss_name = "qal"
    else:
        loss_name = "mse"

    # Check for files
    if not os.path.exists(rf"{model_dir}/{generate_model_file_name(model_name, loss_name, set_name, num_epochs)}.pth"):
        print(f"Model file does not exist: {model_dir}/{generate_model_file_name(model_name, loss_name, set_name, num_epochs)}.pth")
        return
    print(f"Evaluating model: {model_dir}/{generate_model_file_name(model_name, loss_name, set_name, num_epochs)}.pth")


    # 1. Creating dataset
    print("1. Creating dataset")
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
        return

    try:
        model.load_state_dict(torch.load(rf"{model_dir}/{generate_model_file_name(model_name, loss_name, set_name, num_epochs)}.pth"))
    except FileNotFoundError as ex:
        print(f"File error: {ex}")
        return
    print("State data load completed")


    # 5. Criterion
    print("\n5. Creating criterion")
    if is_qal_loss:
        print("Evaluation criterion: QALLoss function")
        criterion_eval = rm.QALLoss()
    else:
        print("Evaluation criterion: MSELoss function")
        criterion_eval = nn.MSELoss()


    def evaluation_print(message: str):
        if show_evaluation:
            print(message)


    # 6. Test and evaluation
    print("\n6. Starting evaluation")
    with torch.no_grad():
        test_loss = []
        correct_predictions = [0 for _ in range(max_acc_round_point + 1)]
        n_samples = 0

        for (rotations, labels) in test_loader:
            rotations = rotations.to(device)
            labels = labels.to(device)
            labels = labels.reshape(labels.shape[0], input_size)
            output = model(rotations)

            test_loss.append(criterion_eval(output, labels).item())

            # Calculating accuracy
            output = output.tolist()
            labels = labels.tolist()
            n_samples += 1

            for i in range(len(labels)):
                evaluation_print(f"\noutput: {output[i]}, expected: {labels[i]}")

                if calculate_accuracy:
                    for acc_i in range(max_acc_round_point + 1):
                        w_acc = int(round(output[i][0], acc_i) == round(labels[i][0], acc_i))
                        i_acc = int(round(output[i][1], acc_i) == round(labels[i][1], acc_i))
                        j_acc = int(round(output[i][2], acc_i) == round(labels[i][2], acc_i))
                        k_acc = int(round(output[i][3], acc_i) == round(labels[i][3], acc_i))

                        current_predictions = (w_acc + i_acc + j_acc + k_acc)
                        correct_predictions[acc_i] += current_predictions
                        evaluation_print(f"accuracy [{acc_i}]: {((100 * correct_predictions[acc_i]) / (input_size)):.2f}%")

            evaluation_print(f"loss: {test_loss[n_samples - 1]}")

        test_loss = np.array(test_loss)
        loss_mean = np.mean(test_loss)
        loss_std = np.std(test_loss)
        print("\nSUMMARY")
        print(f'Loss mean: {loss_mean:.7f}, loss std: {loss_std:.7f}')
        
        if calculate_accuracy:
            for i in range(len(correct_predictions)):
                print(f'accuracy [{i}]: {((100 * correct_predictions[i]) / (input_size * batch_size * n_samples)):.2f}')



if __name__ == "__main__":
    # training_path = r"./data/mockup/large/training_data_hip.csv"
    # labels_path = r"./data/mockup/large/labels_data_hip.csv"
    # training_path = r"./data/mockup/large/training_data_foot.csv"
    # labels_path = r"./data/mockup/large/labels_data_foot.csv"
    training_path = r"./data/mockup/large/training_data_neck.csv"
    labels_path = r"./data/mockup/large/labels_data_neck.csv"

    input_size = 4             # Quaternion
    sequence_length = 100      # Frames
    num_layers = 2

    hidden_size = 128
    num_classes = 4
    num_epochs = 25
    batch_size = 10

    is_qal_loss = False
    show_evaluation = False
    calculate_accuracy = True
    max_acc_round_point = 7

    set_name = "neck"
    model_dir = rf"./models"

    def execute_saved_evaluation(model_type : ModelType, trained_on_qal : bool):
        saved_evaluation(
            input_size = input_size,
            sequence_length = sequence_length,
            num_layers = num_layers, 

            hidden_size = hidden_size, 
            num_classes = num_classes, 
            num_epochs = num_epochs, 
            batch_size = batch_size,
            trained_on_qal = trained_on_qal,

            model_type = model_type,
            is_qal_loss = is_qal_loss,
            show_evaluation = show_evaluation,
            calculate_accuracy = calculate_accuracy,
            max_acc_round_point = max_acc_round_point,

            model_dir = model_dir,
            set_name = set_name,
            training_path = training_path,
            labels_path = labels_path
        )

    execute_saved_evaluation(ModelType.LSTM, False)        # LSTM MSE
    execute_saved_evaluation(ModelType.LSTM, True)         # LSTM QAL
    execute_saved_evaluation(ModelType.QLSTM, False)       # QLSTM MSE
    execute_saved_evaluation(ModelType.QLSTM, True)        # QLSTM QAL