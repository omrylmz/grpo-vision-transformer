import torch


def train_model(model, data_loader, device, optimizer, loss_fn):
    """
    Trains the model on the given data_loader and returns the accuracy.
    """
    model.train()
    running_loss = 0.0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(data_loader.dataset)


def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on the given data_loader and returns the accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
