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

def sample_group_info(current_dist, ref_dist, true_label, group_size):
    """
    For a single example:
      - Sample `group_size` predictions from the current distribution.
      - For each sample, compute:
          * log probability (from current_dist)
          * binary reward: 1.0 if sampled prediction equals true_label, else 0.0
          * probability ratio = current_dist(sample) / (ref_dist(sample) + eps)
    Returns:
      group_log_probs: list of log probabilities (differentiable)
      group_rewards: list of rewards (floats)
      group_ratios: list of ratio tensors (differentiable)
    """
    eps = 1e-8
    group_log_probs = []
    group_rewards = []
    group_ratios = []
    for _ in range(group_size):
        # Sample an action from the current distribution.
        sampled_class = torch.multinomial(current_dist, num_samples=1).item()
        # Compute log probability (differentiable)
        log_prob = torch.log(current_dist[sampled_class] + eps)
        # Compute reward: binary reward (1 if correct, 0 otherwise)
        reward = 1.0 if sampled_class == true_label else 0.0
        # Compute ratio: current probability divided by reference probability.
        ratio = (current_dist[sampled_class] + eps) / (ref_dist[sampled_class] + eps)
        group_log_probs.append(log_prob)
        group_rewards.append(reward)
        group_ratios.append(ratio)
    return group_log_probs, group_rewards, group_ratios


def compute_advantages(group_rewards):
    """
    Given a list of rewards for a group, compute the advantage as the z-score:
        advantage = (reward - mean) / (std + eps)
    Returns a tensor of advantages.
    """
    eps = 1e-8
    rewards_tensor = torch.tensor(group_rewards, dtype=torch.float,
                                  device=group_rewards[0].device if isinstance(group_rewards[0],
                                                                               torch.Tensor) else None)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std() if rewards_tensor.std() > 0 else 1.0
    advantages = (rewards_tensor - mean_reward) / (std_reward + eps)
    return advantages


def compute_surrogate_loss(group_ratios, advantages, clip_eps):
    """
    Computes the surrogate loss for a single example:
      For each sample, calculate:
          L_sample = min( ratio * advantage, clip(ratio, 1-clip_eps, 1+clip_eps) * advantage )
      The loss for the example is the negative average of these values.
    Returns the scalar loss for the example.
    """
    # Convert list of ratios to tensor while preserving gradients.
    ratios_tensor = torch.stack(group_ratios)  # shape: (group_size,)
    clipped_ratios = torch.clamp(ratios_tensor, 1 - clip_eps, 1 + clip_eps)
    surrogate_terms = torch.min(ratios_tensor * advantages, clipped_ratios * advantages)
    return -surrogate_terms.mean()


def grpo_rl_finetuning_loop(model, train_loader, num_rl_iters=3, group_size=5, clip_eps=0.2, beta=0.04, rl_lr=1e-4,
                            device='cuda'):
    """
    GRPO RL Fine-Tuning Loop

    For each image in the batch:
      1. Compute the current model's probability distribution (p_current) and a frozen reference distribution (p_ref).
      2. For each example:
         a. Sample a group of predictions and compute log probabilities, rewards, and probability ratios.
         b. Compute advantages as the z-score of the group rewards.
         c. Compute the surrogate loss using clipping (as in PPO).
      3. Compute the KL divergence penalty between p_current and p_ref.
      4. Sum the surrogate loss and KL penalty to form the total loss.
      5. Backpropagate and update the model.

    Returns a list of average RL losses per RL iteration and plots the RL loss curve.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)

    rl_losses = []
    eps = 1e-8
    for rl_iter in range(num_rl_iters):
        model.train()
        running_loss = 0.0
        sample_count = 0

        # Create a frozen reference model from the current state.
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # Get current distribution
            logits = model(images)  # (batch, num_classes)
            p_current = F.softmax(logits, dim=1)
            with torch.no_grad():
                ref_logits = ref_model(images)
                p_ref = F.softmax(ref_logits, dim=1)

            example_losses = []
            # Process each example in the batch
            for i in range(batch_size):
                current_dist = p_current[i]  # (num_classes,), differentiable
                ref_dist = p_ref[i]  # (num_classes,), constant
                true_label = labels[i].item()

                # Step 1: Sample group information.
                group_log_probs, group_rewards, group_ratios = sample_group_info(
                    current_dist, ref_dist, true_label, group_size
                )
                # Step 2: Compute advantages for the group.
                # (Note: Since rewards are scalars, we convert them to tensor here.)
                advantages = compute_advantages(group_rewards)
                # Step 3: Compute surrogate loss for this example.
                example_loss = compute_surrogate_loss(group_ratios, advantages, clip_eps)
                example_losses.append(example_loss)

            if len(example_losses) == 0:
                continue
            batch_surrogate_loss = torch.stack(example_losses).mean()

            # Step 4: Compute KL divergence between current and reference distributions.
            kl = torch.sum(p_current * (torch.log(p_current + eps) - torch.log(p_ref + eps)), dim=1)
            kl_loss = kl.mean()

            # Step 5: Combine surrogate loss with KL penalty.
            total_loss = batch_surrogate_loss + beta * kl_loss

            # Ensure total_loss depends on model parameters (avoid constant zero gradients).
            if not total_loss.requires_grad:
                dummy = sum(p.sum() for p in model.parameters())
                total_loss = total_loss + 0.0 * dummy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_size
            sample_count += batch_size

        avg_rl_loss = running_loss / sample_count if sample_count > 0 else float('nan')
        rl_losses.append(avg_rl_loss)
        print(f"RL Iteration {rl_iter + 1}/{num_rl_iters}, Avg RL Loss: {avg_rl_loss:.4f}")

    # Plot the RL loss curve.
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_rl_iters + 1), rl_losses, marker='o')
    plt.xlabel("RL Iteration")
    plt.ylabel("Average RL Loss")
    plt.title("GRPO RL Loss Over Iterations")
    plt.show()
    return rl_losses
