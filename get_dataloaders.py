import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as transforms


# -----------------------------
# 2. Data Preparation: Filter CIFAR10 for Cat (label 3) and Dog (label 5)
# -----------------------------

def get_cat_dog_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # Filter for cat (class 3) and dog (class 5)
    def filter_cat_dog(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
        subset = Subset(dataset, indices)
        return subset

    train_subset = filter_cat_dog(trainset)
    test_subset = filter_cat_dog(testset)

    # Remap labels: cat -> 0, dog -> 1
    def remap_labels(batch):
        # batch is a tuple (image, label) from the Subset, so we need to remap label.
        image, label = batch
        new_label = 0 if label == 3 else 1
        return image, new_label

    # Wrap the Subset with a simple collate function that remaps labels
    def collate_fn(batch):
        images, labels = zip(*[remap_labels(item) for item in batch])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader
