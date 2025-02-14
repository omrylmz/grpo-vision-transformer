import torch
from matplotlib import pyplot as plt
from torch import nn as nn


# -----------------------------
# 3. Supervised Training Loop
# -----------------------------

def supervised_training_loop(model, train_loader, test_loader, num_epochs=5, lr=1e-3, device='cuda'):
    """
    Trains the model in a supervised manner using cross-entropy loss and Adam optimizer.
    Also evaluates test accuracy after each epoch and plots the training loss and test accuracy.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluate test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Test Acc = {test_acc:.4f}")

    # Plot training loss and test accuracy
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
