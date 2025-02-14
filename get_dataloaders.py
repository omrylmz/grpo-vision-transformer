import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as transforms


# -----------------------------
# 2. Data Preparation: Filter CIFAR10 for Cat (label 3) and Dog (label 5)
# -----------------------------

def prepare_cat_dog_dataloaders(batch_size=128):
    """
    Loads CIFAR-10 dataset, filters only cat (class 3) and dog (class 5) images,
    remaps labels to 0 (cat) and 1 (dog), and returns DataLoaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def filter_cat_dog(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
        return Subset(dataset, indices)

    train_subset = filter_cat_dog(train_set)
    test_subset = filter_cat_dog(test_set)

    def remap_label(item):
        image, label = item
        new_label = 0 if label == 3 else 1
        return image, new_label

    def collate_fn(batch):
        images, labels = zip(*[remap_label(item) for item in batch])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader