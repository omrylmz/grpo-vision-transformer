import torch

from get_dataloaders import prepare_cat_dog_dataloaders
from grpo_rl_train import grpo_rl_finetuning_generic
from grpo_helpers import default_reward_function
from train_helpers import evaluate_model
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

    initial_acc = evaluate_model(model, train_loader, device)
    print(f"Supervised model accuracy: {initial_acc:.4f}")

    print("Starting GRPO RL fine-tuning...")
    grpo_rl_finetuning_generic(model, train_loader, num_rl_iters=3, group_size=5,
                               clip_eps=0.2, beta=0.04, rl_lr=1e-4, device=device,
                               reward_function=default_reward_function)

    final_acc = evaluate_model(model, train_loader, device)

    print(f"Post-RL model accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
