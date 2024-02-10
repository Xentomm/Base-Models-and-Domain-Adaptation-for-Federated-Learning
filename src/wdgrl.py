"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""

import argparse
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision import models
from tqdm import tqdm, trange
from utils import loop_iterable, set_requires_grad, save_training_plots, EarlyStopper
import os
import dataloader
from sklearn.metrics import f1_score
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyModel(nn.Module):
    """
    Custom neural network model combining feature extraction and classification.
    """
    def __init__(self, sequential1, sequential2):
        super(MyModel, self).__init__()
        self.layer = sequential1
        self.fc = sequential2
       
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor after passing through feature extractor and classifier.
        """
        x1 = self.sequential1(x)
        x2 = self.sequential2(x1)
        return x2

def gradient_penalty(critic, h_s, h_t):
    """
    Computes the gradient penalty for the Wasserstein Distance Guided Representation Learning (WDGRL) framework. 
    
    Args:
        critic: The critic (discriminator) model.
        h_s: Hidden representations of the source data.
        h_t: Hidden representations of the target data.
        
    Returns:
        gradient_penalty: The computed gradient penalty.
    """
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def main(args):
    """
    Main function orchestrating the training process of the WDGRL model.
    
    Args:
        args: Command-line arguments passed to the script.
    """
    source_model = models.resnet50()
    num_classes = 2
    num_features = source_model.fc.in_features
    source_model.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, num_classes)
    )

    source_model.load_state_dict(torch.load(args.MODEL_FILE))
    resnet_without_fc = nn.Sequential(*list(source_model.children())[:-1])
    clf_model = source_model
    source_model_fc=source_model.fc
    source_model = resnet_without_fc
    source_model.to(device)
    source_model_fc.to(device)
    
    critic = nn.Sequential(
        nn.Linear(2048, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)


    wd = os.getcwd().replace("\\", "/")
    source_dir = f"C:/Users/tempo/Desktop/data/{args.SOURCE_DATA}" + "/train_data" 
    target_dir = f"C:/Users/tempo/Desktop/data/{args.TARGET_DATA}" + "/train_data" 
    source_loader = dataloader.load_dataset(source_dir)
    target_loader = dataloader.load_dataset(target_dir)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=0.001)
    clf_optim = torch.optim.Adam(clf_model.parameters(), lr=0.001)
    clf_criterion = nn.MSELoss()

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            (source_x, source_y), (target_x, _) = next(batch_iterator)
            # Train critic
            set_requires_grad(source_model, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            source_y = source_y.unsqueeze(1)
            source_y = source_y.to(device)

            with torch.no_grad():
                h_s = source_model(source_x).data.view(source_x.shape[0], -1)
                h_t = source_model(target_x).data.view(target_x.shape[0], -1)
            for _ in range(args.k_critic):
                gp = gradient_penalty(critic, h_s, h_t)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma*gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                features = resnet_without_fc(x).view(x.shape[0], -1)
                features = features.to(device)
                label_preds = source_model_fc(features[:source_x.shape[0]])
                label_y = source_y

                total_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()
                
            # Train classifier
            set_requires_grad(source_model, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = source_model(target_x).view(target_x.shape[0], -1)


                source_preds = source_model_fc(source_features)
                clf_loss = clf_criterion(source_preds, source_y.float())
                wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_critic)
        mean_accuracy = total_accuracy / (args.iterations*args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}, '
                   f'critic_accuracy={mean_accuracy:.4f}')


        modules_all = list(resnet_without_fc.children())# delete the last fc layer.
        modules_fc = list(source_model_fc.children())

        model_Adrian = MyModel(nn.Sequential(*modules_all), nn.Sequential(*modules_fc))
        czysty_resnet = models.resnet50()
        num_classes = 2
        num_features = czysty_resnet.fc.in_features
        czysty_resnet.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, num_classes)
        )
            
        names=[]
        params=[]

        for name,param in czysty_resnet.state_dict().items():
            names.append(name)

        for name_ad, param_ad in model_Adrian.state_dict().items():
            params.append(param_ad)

        for idx in range(len(czysty_resnet.state_dict().items())-1):
            czysty_resnet.state_dict()[names[idx]].copy_(torch.Tensor(params[idx]))
        
        save_path_model = wd + f"/trained_models/adda_{args.SOURCE_DATA}_{args.TARGET_DATA}.pt"

        torch.save(czysty_resnet, save_path_model)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('SOURCE_DATA')
    arg_parser.add_argument('TARGET_DATA')
    arg_parser.add_argument('--batch-size', type=int, default=16)
    arg_parser.add_argument('--iterations', type=int, default=200)
    arg_parser.add_argument('--epochs', type=int, default=5)
    arg_parser.add_argument('--k-critic', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=10)
    arg_parser.add_argument('--gamma', type=float, default=10)
    arg_parser.add_argument('--wd-clf', type=float, default=1)

    args = arg_parser.parse_args()
    main(args)
