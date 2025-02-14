import torch

from get_dataloaders import prepare_cat_dog_dataloaders
from grpo_rl_train import grpo_rl_finetuning_generic
from helper import default_reward_function
from supervised_train import supervised_training
from vision_transformer import VisionTransformerClassifier


# -----------------------------
# 5. Main: Supervised Training then GRPO RL Fine-Tuning
# -----------------------------

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    batch_size = 128
    num_supervised_epochs = 5
    supervised_lr = 1e-3

    # Use the ViT application: cat vs. dog classification.
    train_loader, test_loader = prepare_cat_dog_dataloaders(batch_size=batch_size)

    model = VisionTransformerClassifier(image_size=32, patch_size=4, in_channels=3,
                                        embed_dim=64, num_heads=4, num_layers=2, num_classes=2)

    print("Starting supervised training...")
    supervised_training(model, train_loader, test_loader, num_epochs=num_supervised_epochs,
                        lr=supervised_lr, device=device)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    initial_acc = correct / total
    print(f"Supervised model accuracy: {initial_acc:.4f}")

    print("Starting GRPO RL fine-tuning...")
    grpo_rl_finetuning_generic(model, train_loader, num_rl_iters=3, group_size=5,
                               clip_eps=0.2, beta=0.04, rl_lr=1e-4, device=device,
                               reward_function=default_reward_function)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    final_acc = correct / total
    print(f"Post-RL model accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
