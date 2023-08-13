import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

import recurrent_models as rm
from dataset_initializer import RotationDataset

def saved_evaluation(
        input_size = 4,          # Quaternion
        sequence_length = 100,   # Frames
        batch_size = 10,

        is_qal_loss = False,
        show_evaluation = False,
        calculate_accuracy = True,
        max_acc_round_point = 7,

        model_path = "",
        training_path = r"./data/mockup/large/training_data.csv",
        labels_path = r"./data/mockup/large/labels_data.csv"
):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Saved evaluation procedure started")


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
    print("4. Reading model")
    try:
        model = torch.load(model_path).to(device)
    except FileNotFoundError:
        print(f"Model file {model_path} doesn't exist")
        sys.exit()
    print(f"Model type: {type(model)}")


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
    # model_path = rf"./models/lstm_mse_batch10_epochs3.pth"
    # model_path = rf"./models/lstm_qal_batch10_epochs3.pth"
    # model_path = rf"./models/qlstm_mse_batch10_epochs3.pth"
    # model_path = rf"./models/qlstm_qal_batch10_epochs3.pth"
    # model_path = rf"./models/VectorizedQLSTM_mse_batch10_epochs3.pth"
    # model_path = rf"./models/VectorizedQLSTM_qal_batch10_epochs3.pth"

    # training_path = r"./data/mockup/training_data (Medium).csv"
    # labels_path = r"./data/mockup/labels_data (Medium).csv"
    training_path = r"./data/mockup/large/training_data.csv"
    labels_path = r"./data/mockup/large/labels_data.csv"

    input_size = 4          # Quaternion
    sequence_length = 100   # Frames
    batch_size = 10

    is_qal_loss = True
    show_evaluation = False
    calculate_accuracy = True
    max_acc_round_point = 7

    # LSTM MSE
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/lstm_mse_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )

    # LSTM QAL
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/lstm_qal_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )

    # QLSTM MSE
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/qlstm_mse_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )

    # QLSTM QAL
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/qlstm_qal_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )

    # Vectorized QLSTM MSE
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/VectorizedQLSTM_mse_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )

    # Vectorized QLSTM QAL
    saved_evaluation(
        input_size = input_size,
        sequence_length = sequence_length,
        batch_size = batch_size,

        is_qal_loss = is_qal_loss,
        show_evaluation = show_evaluation,
        calculate_accuracy = calculate_accuracy,
        max_acc_round_point = max_acc_round_point,

        model_path = rf"./models/VectorizedQLSTM_qal_batch10_epochs3.pth",
        training_path = training_path,
        labels_path = labels_path
    )