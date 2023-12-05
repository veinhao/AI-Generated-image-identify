import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from data.dataset import CustomImageDataset
from train.util.printf import printf


def create_data_loaders(config):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        # ... any additional preprocessing steps
    ])

    # Split the dataset into training and validation sets
    train_size = None
    val_size = None
    train_loader = None
    val_loader = None
    if 1 > config.VALIDATION_SPLIT > 0:

        # Create the dataset
        dataset = CustomImageDataset(config.DATAPATH, transform=transform)
        printf('train and val')

        train_size = int((1 - config.VALIDATION_SPLIT) * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(config.RANDOM_SEED))

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=config.SHUFFLE)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                                shuffle=False)
        return train_loader, val_loader
    elif config.VALIDATION_SPLIT == 1:

        printf('only val')

        val_dataset = CustomImageDataset(config.DATAPATH, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                                shuffle=False)
        return val_loader
    elif config.VALIDATION_SPLIT == 0:

        printf('only train')

        train_dataset = CustomImageDataset(config.DATAPATH, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=config.SHUFFLE)
        return train_loader

    if config.VALIDATION_SPLIT == -1:
        # test_dataset = CustomImageDataset(config.DATAPATH, transform=transform)
        # test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
        #                          shuffle=False)

        return None
