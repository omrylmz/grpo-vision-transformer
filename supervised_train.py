import torch
from matplotlib import pyplot as plt
from torch import nn as nn

from train_helpers import train_model, evaluate_model


# -----------------------------
# 3. Supervised Training Loop
# -----------------------------

def supervised_training(model, train_loader, test_loader, num_epochs=5, lr=1e-3, device='cuda'):
    """
    Supervised training loop using cross-entropy loss.
    Returns lists of training losses and test accuracies per epoch.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_accuracies = [], []
    for epoch in range(num_epochs):
        epoch_loss = train_model(model, train_loader, device, optimizer, criterion)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Test Acc = {test_acc:.4f}")

    # Plot training loss and test accuracy.
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Supervised Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy")
    plt.tight_layout()
    plt.show()
    return train_losses, test_accuracies
