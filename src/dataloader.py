import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_dataset(data_dir, batch_size=64, shuffle=True):
    """
    Load a dataset using PyTorch's ImageFolder.

    Parameters:
    - data_dir: Path to the dataset directory.
    - batch_size: Batch size for the data loader.
    - shuffle: Whether to shuffle the data during training.

    Returns:
    - A DataLoader object for the specified dataset.
    """
    # Define the transformations for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)

    return data_loader