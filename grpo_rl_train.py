import copy

import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F


# -----------------------------
# 4. GRPO RL Training Loop
# -----------------------------
# Here we implement GRPO (Group Relative Policy Optimization) on top of our trained model.
# For each image in a batch, we will sample multiple predictions (from the model's softmax),
# compute a binary reward (1 if sampled prediction == ground truth, else 0),
# compute the probability ratios with respect to a reference model,
# clip the ratios, add a KL divergence penalty, and backpropagate.

def grpo_rl_train(model, train_loader, num_rl_iters=3, group_size=5, clip_eps=0.2, beta=0.04, rl_lr=1e-4,
                  device='cuda'):
    """
    Run GRPO reinforcement learning on the model.

    For each example in a batch:
      - Compute current model distribution (p_current) over classes.
      - Using p_current, sample group_size predictions.
      - Compute binary reward: 1 if sampled prediction equals ground truth, else 0.
      - For each sample, compute the ratio = p_current(sample) / (p_ref(sample) + eps)
        (with p_ref from a frozen reference model).
      - Compute advantage for the group as the z-score: (reward - mean_reward) / (std_reward + eps).
      - Compute the surrogate loss using the ratio (with clipping) multiplied by the advantage.
      - Add a KL divergence penalty between the current and reference distributions.
      - Backpropagate and update the model.

    The reference model is a frozen copy of the model at the beginning of each RL iteration.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)

    rl_losses = []
    for rl_iter in range(num_rl_iters):
        model.train()
        running_loss = 0.0
        count = 0

        # Create a reference model (frozen copy)
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            logits = model(images)  # (batch, num_classes)
            p_current = F.softmax(logits, dim=1)  # (batch, num_classes)
            with torch.no_grad():
                ref_logits = ref_model(images)  # (batch, num_classes)
                p_ref = F.softmax(ref_logits, dim=1)  # (batch, num_classes)

            sample_loss_list = []
            # Process each example in the batch
            for i in range(batch_size):
                dist = p_current[i]  # (num_classes,), requires grad
                ref_dist = p_ref[i]  # (num_classes,)
                true_label = labels[i].item()

                group_log_probs = []
                group_rewards = []
                group_ratios = []
                for _ in range(group_size):
                    # Sample an action from the current distribution.
                    sampled_class = torch.multinomial(dist, num_samples=1).item()
                    log_prob = torch.log(dist[sampled_class] + 1e-8)  # differentiable
                    reward = 1.0 if sampled_class == true_label else 0.0  # binary reward (mock)
                    ratio = (dist[sampled_class] + 1e-8) / (ref_dist[sampled_class] + 1e-8)

                    group_log_probs.append(log_prob)
                    group_rewards.append(reward)
                    group_ratios.append(ratio)  # keep as tensor (via stacking later)

                # Stack group_ratios to preserve gradients.
                group_ratios = torch.stack(group_ratios)  # (group_size,)
                # Create a tensor for rewards (non-differentiable; treated as constant)
                group_rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float)
                mean_reward = group_rewards_tensor.mean()
                std_reward = group_rewards_tensor.std() if group_rewards_tensor.std() > 0 else 1.0
                advantages = (group_rewards_tensor - mean_reward) / (std_reward + 1e-8)

                clipped_ratio = torch.clamp(group_ratios, 1 - clip_eps, 1 + clip_eps)
                surrogate = torch.min(group_ratios * advantages, clipped_ratio * advantages)
                example_loss = - surrogate.mean()
                sample_loss_list.append(example_loss)

            if len(sample_loss_list) == 0:
                continue
            batch_loss = torch.stack(sample_loss_list).mean()

            # Compute KL divergence between current and reference distributions.
            kl = torch.sum(p_current * (torch.log(p_current + 1e-8) - torch.log(p_ref + 1e-8)), dim=1)
            kl_loss = kl.mean()

            total_loss = batch_loss + beta * kl_loss
            # --- Prevent total_loss from being a constant with no grad ---
            if not total_loss.requires_grad:
                dummy = sum(p.sum() for p in model.parameters())
                total_loss = total_loss + 0.0 * dummy
            # ---------------------------------------------------------------

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_size
            count += batch_size

        avg_rl_loss = running_loss / count if count > 0 else float('nan')
        rl_losses.append(avg_rl_loss)
        print(f"RL Iteration {rl_iter + 1}/{num_rl_iters}, RL Loss: {avg_rl_loss:.4f}")

    # Plot the RL loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_rl_iters + 1), rl_losses, marker='o')
    plt.xlabel("RL Iteration")
    plt.ylabel("Average RL Loss")
    plt.title("GRPO RL Loss Over Iterations")
    plt.show()


def compute_kl_divergence(p_current, p_ref):
    # p_current, p_ref: tensors of shape (batch, num_classes)
    # We compute KL(p_current || p_ref) for each sample.
    # Use log probabilities of p_current and treat p_ref as target.
    # F.kl_div expects log_target as input? Actually, F.kl_div(input, target, reduction='batchmean') computes:
    # sum(target * (log(target) - input)). We want KL(p_current || p_ref) = sum p_current * (log p_current - log p_ref)
    # We can compute that manually.
    kl = torch.sum(p_current * (torch.log(p_current + 1e-8) - torch.log(p_ref + 1e-8)), dim=1)
    return kl.mean()
