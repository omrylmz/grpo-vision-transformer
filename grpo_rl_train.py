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

def grpo_rl_finetuning_loop(model, train_loader, num_rl_iters=3, group_size=5, clip_eps=0.2, beta=0.04, rl_lr=1e-4,
                            device='cuda'):
    """
    Applies GRPO (Group Relative Policy Optimization) for RL fine-tuning.

    For each image in the batch:
      - Obtain current probability distribution over classes.
      - Use a frozen reference model to compute reference probabilities.
      - For each example, sample a group of predictions and compute:
           - Log probabilities from the current model.
           - A binary reward (1 if prediction equals ground truth, else 0).
           - The ratio of current probability to reference probability.
      - Compute the advantage as the z-score of rewards within the group.
      - Compute the surrogate loss with clipping (as in PPO).
      - Add a KL divergence penalty between current and reference distributions.
      - Backpropagate the loss and update the model.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)

    rl_losses = []
    for rl_iter in range(num_rl_iters):
        model.train()
        running_loss = 0.0
        sample_count = 0

        # Create a frozen reference model from current model state.
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            logits = model(images)  # (batch, num_classes)
            p_current = F.softmax(logits, dim=1)  # (batch, num_classes)
            with torch.no_grad():
                ref_logits = ref_model(images)  # (batch, num_classes)
                p_ref = F.softmax(ref_logits, dim=1)

            example_losses = []
            # Process each example individually.
            for i in range(batch_size):
                current_dist = p_current[i]  # (num_classes,) - differentiable
                ref_dist = p_ref[i]  # (num_classes,) - constant
                true_label = labels[i].item()

                group_log_probs = []
                group_rewards = []
                group_ratios = []
                for _ in range(group_size):
                    # Sample a prediction from current distribution.
                    sampled_class = torch.multinomial(current_dist, num_samples=1).item()
                    log_prob = torch.log(current_dist[sampled_class] + 1e-8)
                    reward = 1.0 if sampled_class == true_label else 0.0
                    ratio = (current_dist[sampled_class] + 1e-8) / (ref_dist[sampled_class] + 1e-8)

                    group_log_probs.append(log_prob)
                    group_rewards.append(reward)
                    group_ratios.append(ratio)  # Preserve grad by stacking later

                # Stack ratios to maintain gradient flow.
                group_ratios = torch.stack(group_ratios)  # shape: (group_size,)
                rewards_tensor = torch.tensor(group_rewards, device=device, dtype=torch.float)
                mean_reward = rewards_tensor.mean()
                std_reward = rewards_tensor.std() if rewards_tensor.std() > 0 else 1.0
                advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)

                clipped_ratio = torch.clamp(group_ratios, 1 - clip_eps, 1 + clip_eps)
                surrogate = torch.min(group_ratios * advantages, clipped_ratio * advantages)
                example_loss = - surrogate.mean()
                example_losses.append(example_loss)

            if len(example_losses) == 0:
                continue
            batch_loss = torch.stack(example_losses).mean()
            # Compute average KL divergence over the batch.
            kl = torch.sum(p_current * (torch.log(p_current + 1e-8) - torch.log(p_ref + 1e-8)), dim=1)
            kl_loss = kl.mean()

            total_loss = batch_loss + beta * kl_loss

            # Ensure total_loss depends on model parameters even if constant.
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
