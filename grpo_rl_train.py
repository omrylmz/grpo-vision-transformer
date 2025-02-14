import copy

import torch
from matplotlib import pyplot as plt

from grpo_helpers import get_probability_distribution, sample_group_info, compute_advantages, compute_surrogate_loss, \
    compute_kl_divergence, default_reward_function


# -----------------------------
# 4. GRPO RL Training Loop
# -----------------------------
# Here we implement GRPO (Group Relative Policy Optimization) on top of our trained model.
# For each image in a batch, we will sample multiple predictions (from the model's softmax),
# compute a binary reward (1 if sampled prediction == ground truth, else 0),
# compute the probability ratios with respect to a reference model,
# clip the ratios, add a KL divergence penalty, and backpropagate.


def grpo_rl_finetuning_generic(model, data_loader, num_rl_iters=3, group_size=5, clip_eps=0.2,
                               beta=0.04, rl_lr=1e-4, device='cuda', reward_function=default_reward_function):
    """
    Generic GRPO RL Fine-Tuning Loop that can be transferred to another transformer-based application.

    Steps:
      1. For each batch, compute current probability distribution (p_current) and obtain a frozen reference distribution (p_ref).
      2. For each example in the batch, sample a group of predictions and compute:
            - Log probabilities, rewards, and probability ratios (via a reward function).
      3. Compute advantages as the z-score of rewards.
      4. Compute the surrogate loss (using clipping) per example.
      5. Compute KL divergence penalty between current and reference distributions.
      6. Combine surrogate loss and KL penalty to form total loss, backpropagate, and update the model.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)
    rl_losses = []
    eps = 1e-8

    for rl_iter in range(num_rl_iters):
        model.train()
        running_loss = 0.0
        total_samples = 0

        # Create a frozen reference model from current state.
        ref_model = copy.deepcopy(model)
        ref_model.eval()

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            p_current = get_probability_distribution(model, inputs, device)  # (batch, num_classes)
            with torch.no_grad():
                p_ref = get_probability_distribution(ref_model, inputs, device)

            example_losses = []
            for i in range(batch_size):
                current_dist = p_current[i]  # (num_classes,)
                ref_dist = p_ref[i]
                true_label = labels[i].item()

                group_log_probs, group_rewards, group_ratios = sample_group_info(
                    current_dist, ref_dist, true_label, group_size, reward_function
                )
                advantages = compute_advantages(group_rewards, device=current_dist.device)
                example_loss = compute_surrogate_loss(group_ratios, advantages, clip_eps)
                example_losses.append(example_loss)

            if len(example_losses) == 0:
                continue
            batch_surrogate_loss = torch.stack(example_losses).mean()
            kl_loss = compute_kl_divergence(p_current, p_ref, eps)
            total_loss = batch_surrogate_loss + beta * kl_loss

            # Ensure total_loss depends on model parameters.
            if not total_loss.requires_grad:
                dummy = sum(p.sum() for p in model.parameters())
                total_loss = total_loss + 0.0 * dummy

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_size
            total_samples += batch_size

        avg_rl_loss = running_loss / total_samples if total_samples > 0 else float('nan')
        rl_losses.append(avg_rl_loss)
        print(f"RL Iteration {rl_iter + 1}/{num_rl_iters}, Avg RL Loss: {avg_rl_loss:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_rl_iters + 1), rl_losses, marker='o')
    plt.xlabel("RL Iteration")
    plt.ylabel("Average RL Loss")
    plt.title("GRPO RL Loss Over Iterations")
    plt.show()

    return rl_losses
