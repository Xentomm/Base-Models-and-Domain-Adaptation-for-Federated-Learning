"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from utils import GradientReversal, loop_iterable, set_requires_grad
from tqdm import tqdm, trange
import dataloader
import os

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
    Main function orchestrating the training process of the RevGrad model.
    
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
    
    source_model_fc=source_model.fc
    source_model_fc = source_model_fc.to(device)
    source_model = resnet_without_fc
    clf = source_model
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
        GradientReversal(),
        nn.Linear(2048, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
 
    # Load datasets
    wd = os.getcwd().replace("\\", "/")
    source_dir = f"C:/Users/tempo/Desktop/data/{args.SOURCE_DATA}" + "/train_data" 
    target_dir = f"C:/Users/tempo/Desktop/data/{args.TARGET_DATA}" + "/train_data" 
    source_loader = dataloader.load_dataset(source_dir)
    target_loader = dataloader.load_dataset(target_dir)

    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.01)
    target_optim = torch.optim.Adam(target_model.parameters(), lr=0.01)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(source_model.parameters()), lr=0.01)

    # Training loop
    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_label_accuracy = 0
        for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
                x = torch.cat([source_x, target_x])
                x = x.to(device)
                source_x = source_x.to(device)
                target_x = target_x.to(device)
                domain_y = torch.cat([torch.ones(source_x.shape[0]),
                                      torch.zeros(target_x.shape[0])])
                domain_y = domain_y.to(device)
                label_y = source_labels.to(device)

                features = resnet_without_fc(x).view(x.shape[0], -1)
                features = features.to(device)
                domain_preds = discriminator(features).squeeze()
                label_preds = source_model_fc(features[:source_x.shape[0]])
                
                domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
                label_loss = F.cross_entropy(label_preds, label_y)
                loss = domain_loss + label_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_domain_loss += domain_loss.item()
                total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

        mean_loss = total_domain_loss / n_batches
        mean_accuracy = total_label_accuracy / n_batches
        tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
                   f'source_accuracy={mean_accuracy:.4f}')

        # Save the trained model
        model_Adrian = MyModel(nn.Sequential(*list(resnet_without_fc.children())), nn.Sequential(*list(source_model_fc.children())))
        czysty_resnet = models.resnet50()
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

        save_path_model = wd + f"/trained_models/revgrad_{args.SOURCE_DATA}_{args.TARGET_DATA}.pt"
        torch.save(czysty_resnet, save_path_model)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
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
