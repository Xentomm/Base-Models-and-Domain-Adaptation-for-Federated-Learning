import torch
from torch.autograd import Function
import matplotlib.pyplot as plt

def set_requires_grad(model, requires_grad=True):
    """
    Function to set requires_grad attribute for all parameters of a model.

    Args:
        model (torch.nn.Module): The model for which parameters' requires_grad attribute will be set.
        requires_grad (bool): Whether to set requires_grad to True or False.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    """
    Function to infinitely loop through an iterable.

    Args:
        iterable: Any iterable object.

    Yields:
        Elements of the iterable in an infinite loop.
    """
    while True:
        yield from iterable

def save_training_plots(train_losses, train_accuracies, train_f1_scores, 
                        val_losses, val_accuracies, val_f1_scores, 
                        save_path):
    """
    Function to save training plots.

    Args:
        train_losses (list): List of training losses for each epoch.
        train_accuracies (list): List of training accuracies for each epoch.
        train_f1_scores (list): List of training F1 scores for each epoch.
        val_losses (list): List of validation losses for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        val_f1_scores (list): List of validation F1 scores for each epoch.
        save_path (str): Path to save the plots.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plotting training, validation, and test losses
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation Losses')
    plt.legend()
    plt.savefig(save_path + '_loss.png')
    plt.close()

    # Plotting training, validation, and test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation Accuracies')
    plt.legend()
    plt.savefig(save_path + '_accuracy.png')
    plt.close()

    # Plotting training, validation, and test F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1_scores, label='Training F1 Score')
    plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training, Validation F1 Scores')
    plt.legend()
    plt.savefig(save_path + '_f1_score.png')
    plt.close()

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """
        Early stopper to monitor validation loss for early stopping.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Check if early stopping criterion is met.

        Args:
            validation_loss (float): Current validation loss.

        Returns:
            bool: True if early stopping criterion is met, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
