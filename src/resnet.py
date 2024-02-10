import argparse
import os
import torch
import torch.nn as nn
import dataloader
from torchvision import models
from sklearn.metrics import f1_score, confusion_matrix
from utils import save_training_plots, EarlyStopper
import sys

def main(args):
    """
    Main function for training a ResNet50 model and evaluating it on different datasets.

    Args:
        args: Command-line arguments passed to the script.
    """
    wd = os.getcwd().replace("\\", "/")
    save_path_model = wd + f"/trained_models/{args.FOLDER_NAME}_{args.TRAINING_DATA}"
    os.makedirs(save_path_model, exist_ok=True)
    # logging file
    if args.debug == "False":
        f = open(save_path_model + f"/{args.FOLDER_NAME}.txt" , 'w')
        sys.stdout = f
    print(f"================== Resnet50 ==================")
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    train_data_dir = f"C:/Users/tempo/Desktop/data/{args.TRAINING_DATA}" + "/train_data" 
    val_data_dir = f"C:/Users/tempo/Desktop/data/{args.TRAINING_DATA}" + "/validation_data"

    train_loader = dataloader.load_dataset(train_data_dir)
    val_loader = dataloader.load_dataset(val_data_dir)

    print(f"================== Train {args.TRAINING_DATA} ==================")

    resnet50 = models.resnet50(weights='DEFAULT')
    num_classes = 2

    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, num_classes)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    resnet50.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=False)

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    for epoch in range(1, args.epochs+1):
        # Training loop
        resnet50.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        true_labels = []
        pred_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        epoch_f1 = f1_score(true_labels, pred_labels, average='weighted')

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_f1_scores.append(epoch_f1)
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, F1 Score: {epoch_f1:.4f}')

        # Validation loop
        resnet50.eval()
        val_correct_predictions = 0
        val_total_samples = 0
        val_running_loss = 0.0
        val_true_labels = []
        val_pred_labels = []

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = resnet50(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                scheduler.step(val_running_loss)

                _, val_predicted = torch.max(val_outputs, 1)
                val_total_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

                val_true_labels.extend(val_labels.cpu().numpy())
                val_pred_labels.extend(val_predicted.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        val_epoch_f1 = f1_score(val_true_labels, val_pred_labels, average='weighted')

        if early_stopper.early_stop(val_epoch_loss):             
            break

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        val_f1_scores.append(val_epoch_f1)

        print(f'Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy * 100:.2f}%, F1 Score: {val_epoch_f1:.4f}')

        # Save model
        torch.save(resnet50.state_dict(), save_path_model + f"/resnet50_{args.TRAINING_DATA}.pt")

    # Test loop rsna
    test_data_dir = "C:/Users/tempo/Desktop/data/rsna/test_data"
    test_loader = dataloader.load_dataset(test_data_dir)
    resnet50.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = resnet50(test_inputs)
            test_loss = criterion(test_outputs, test_labels)

            test_running_loss += test_loss.item()

            _, test_predicted = torch.max(test_outputs, 1)
            test_total_samples += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

            test_true_labels.extend(test_labels.cpu().numpy())
            test_pred_labels.extend(test_predicted.cpu().numpy())
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)

    # Extract TP, TN, FP, FN
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_accuracy = test_correct_predictions / test_total_samples
    test_epoch_f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')

    print("================== Test RSNA ==================")
    print(f'Test Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_accuracy * 100:.2f}%, F1 Score: {test_epoch_f1:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

    # Test loop vindir
    test_data_dir = "C:/Users/tempo/Desktop/data/vindir/test_data"
    test_loader = dataloader.load_dataset(test_data_dir)
    resnet50.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = resnet50(test_inputs)
            test_loss = criterion(test_outputs, test_labels)

            test_running_loss += test_loss.item()

            _, test_predicted = torch.max(test_outputs, 1)
            test_total_samples += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

            test_true_labels.extend(test_labels.cpu().numpy())
            test_pred_labels.extend(test_predicted.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)

    # Extract TP, TN, FP, FN
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_accuracy = test_correct_predictions / test_total_samples
    test_epoch_f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')

    print("================== Test VINDIR ==================")
    print(f'Test Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_accuracy * 100:.2f}%, F1 Score: {test_epoch_f1:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

    # Test loop ddsm
    test_data_dir = "C:/Users/tempo/Desktop/data/ddsm/test_data"
    test_loader = dataloader.load_dataset(test_data_dir)
    resnet50.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = resnet50(test_inputs)
            test_loss = criterion(test_outputs, test_labels)

            test_running_loss += test_loss.item()

            _, test_predicted = torch.max(test_outputs, 1)
            test_total_samples += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

            test_true_labels.extend(test_labels.cpu().numpy())
            test_pred_labels.extend(test_predicted.cpu().numpy())

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)

    # Extract TP, TN, FP, FN
    TP = conf_matrix[1, 1]  # True Positives
    TN = conf_matrix[0, 0]  # True Negatives
    FP = conf_matrix[0, 1]  # False Positives
    FN = conf_matrix[1, 0]  # False Negatives

    # Calculate Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_accuracy = test_correct_predictions / test_total_samples
    test_epoch_f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')

    print("================== Test DDSM ==================")
    print(f'Test Loss: {test_epoch_loss:.4f}, Accuracy: {test_epoch_accuracy * 100:.2f}%, F1 Score: {test_epoch_f1:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

    if args.debug == "False":
        f.close() # close logging file
    
    save_training_plots(train_losses, train_accuracies, train_f1_scores,
                        val_losses, val_accuracies, val_f1_scores,
                        save_path_model + "/")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Resnet training')
    arg_parser.add_argument('TRAINING_DATA', help='Name of the training data')
    arg_parser.add_argument('FOLDER_NAME', help='Name of the folder for plots and save model')
    arg_parser.add_argument('--epochs', type=int, default=20)
    arg_parser.add_argument("--debug", default=True, help="debug mode") 
    args = arg_parser.parse_args()
    main(args)
