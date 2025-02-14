import argparse

import torch

from get_dataloaders import prepare_dataloaders
from grpo_rl_train import grpo_rl_finetuning_generic
from grpo_helpers import default_reward_function
from train_helpers import evaluate_model
from supervised_train import supervised_training
from vision_transformer import VisionTransformerClassifier


# -----------------------------
# 5. Main: Supervised Training then GRPO RL Fine-Tuning
# -----------------------------

def parse_args():
    """
    Parses command line arguments and returns them.
    """
    parser = argparse.ArgumentParser(description="GRPO RL Fine-Tuning Demo with ViT")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_supervised_epochs", type=int, default=5, help="Number of supervised training epochs")
    parser.add_argument("--supervised_lr", type=float, default=1e-3, help="Learning rate for supervised training")
    parser.add_argument("--num_rl_iters", type=int, default=3, help="Number of GRPO RL iterations")
    parser.add_argument("--group_size", type=int, default=5, help="Group size for GRPO sampling")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="Clipping epsilon for GRPO")
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence penalty weight")
    parser.add_argument("--rl_lr", type=float, default=1e-4, help="Learning rate for RL fine-tuning")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: cuda, mps, or cpu (auto-selected if not provided)")
    args = parser.parse_args()
    return (args.batch_size, args.num_supervised_epochs, args.supervised_lr, args.num_rl_iters, args.group_size, args.clip_eps,
            args.beta, args.rl_lr, args.device)


def main():
    batch_size, num_supervised_epochs, supervised_lr, num_rl_iters, group_size, clip_eps, beta, rl_lr, device = parse_args()

    # Device selection
    if not device:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Set hyperparameters from arguments.

    # Prepare DataLoaders (for example, cat vs. dog classification)
    train_loader, test_loader = prepare_dataloaders(batch_size=batch_size)

    # Instantiate the Vision Transformer classifier.
    model = VisionTransformerClassifier(
        image_size=32, patch_size=4, in_channels=3,
        embed_dim=64, num_heads=4, num_layers=2, num_classes=2
    )

    print("Starting supervised training...")
    supervised_training(model, train_loader, test_loader, num_epochs=num_supervised_epochs, lr=supervised_lr, device=device)

    initial_acc = evaluate_model(model, test_loader, device)
    print(f"Supervised model accuracy: {initial_acc:.4f}")

    print("Starting GRPO RL fine-tuning...")
    grpo_rl_finetuning_generic(
        model, train_loader, num_rl_iters=num_rl_iters, group_size=group_size,
        clip_eps=clip_eps, beta=beta, rl_lr=rl_lr, device=device,
        reward_function=default_reward_function
    )

    final_acc = evaluate_model(model, test_loader, device)
    print(f"Post-RL model accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()