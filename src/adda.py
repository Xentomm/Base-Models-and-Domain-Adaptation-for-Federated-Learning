"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse
import torch
from torch import nn
from tqdm import tqdm, trange
from utils import loop_iterable, set_requires_grad
from torchvision import models
import dataloader
import os
from efficientnet_pytorch import EfficientNet

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


def main(args):
    """
    Main function orchestrating the training process of the ADDA model.
    
    Args:
        args: Command-line arguments passed to the script.
    """
    # Load source and target models
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
    set_requires_grad(source_model, requires_grad=False)
    resnet_without_fc = nn.Sequential(*list(source_model.children())[:-1])
    clf = source_model
    source_model_fc=source_model.fc
    source_model = resnet_without_fc
    source_model.to(device)
    target_model = models.resnet50()
    target_model.fc = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(16, num_classes)
    )
    target_model.load_state_dict(torch.load(args.MODEL_FILE))
    resnet_without_fc_target = nn.Sequential(*list(target_model.children())[:-1])
    target_model = resnet_without_fc_target
    target_model.to(device)
    
    discriminator = nn.Sequential(
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

    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.01)
    target_optim = torch.optim.Adam(target_model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])
                
                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_accuracy = total_accuracy / (args.iterations*args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, '
                   f'discriminator_accuracy={mean_accuracy:.4f}')


        modules_all = list(resnet_without_fc.children())# delete the last fc layer.
        modules_fc = list(source_model_fc.children())

        model = MyModel(nn.Sequential(*modules_all), nn.Sequential(*modules_fc))
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

        for name_ad, param_ad in model.state_dict().items():
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
    arg_parser.add_argument('--iterations', type=int, default=5)
    arg_parser.add_argument('--epochs', type=int, default=1)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=1)
    args = arg_parser.parse_args()
    main(args)
