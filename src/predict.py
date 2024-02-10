import argparse
import torch
from tqdm import tqdm
import dataloader
from torchvision import models
from torch import nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from torchvision.models import resnet18, resnet50,efficientnet_b0
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    """
    Main function for testing a trained model on different datasets.
    
    Args:
        args: Command-line arguments passed to the script.
    """
    criterion = nn.CrossEntropyLoss()
    
    # path='C:/Users/tempo/Desktop/pb/Projekt-Badawczy/trained_models/resnet_rsna/resnet50_rsna.pt'
    path='C:/Users/tempo/Desktop/pb/Projekt-Badawczy/trained_models/resnet_vindir/resnet50_vindir.pt'

    resnet = models.resnet50(ResNet50_Weights.DEFAULT)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
    nn.Linear(num_features, 16),
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.Dropout(p=0.6),
    nn.Linear(16, 2)
    )
    model = resnet
    model.load_state_dict(torch.load(path))
    model = model.to(device)

    # Test loop rsna
    test_data_dir = "C:/Users/tempo/Desktop/data/rsna/test_data"
    test_loader = dataloader.load_dataset(test_data_dir)
    model.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []
    # target_names = ['no_cancer', 'cancer']
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
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
    model.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
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
    model.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Test a model')
    # arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    args = arg_parser.parse_args()
    main(args)
