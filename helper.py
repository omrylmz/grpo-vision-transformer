import torch

import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 4. Helper Function: KL Divergence Computation
# -----------------------------------------------------------------------------

def extract_logits(model_output):
    """
    Generic helper to extract logits from model output.
    Assumes model returns a tensor or a dict with key 'logits'.
    """
    if isinstance(model_output, torch.Tensor):
        return model_output
    elif isinstance(model_output, dict) and 'logits' in model_output:
        return model_output['logits']
    else:
        raise ValueError("Unknown model output format!")

def get_probability_distribution(model, inputs, device='cuda'):
    """
    Passes inputs through the model and returns the softmax probability distribution.
    """
    model_output = model(inputs.to(device))
    logits = extract_logits(model_output)
    return F.softmax(logits, dim=1)

def default_reward_function(sampled_class, true_label):
    """
    Default reward function:
      Returns 1.0 if the sampled class equals the true label, else 0.0.
    This can be replaced with a custom reward design.
    """
    return 1.0 if sampled_class == true_label else 0.0

def sample_group_info(current_dist, ref_dist, true_label, group_size, reward_function=default_reward_function):
    """
    For one example, sample 'group_size' predictions from the current distribution.
    Returns:
      - group_log_probs: list of log probabilities (differentiable)
      - group_rewards: list of rewards (floats)
      - group_ratios: list of probability ratios (differentiable)
    """
    eps = 1e-8
    group_log_probs = []
    group_rewards = []
    group_ratios = []
    for _ in range(group_size):
        sampled_class = torch.multinomial(current_dist, num_samples=1).item()
        log_prob = torch.log(current_dist[sampled_class] + eps)
        reward = reward_function(sampled_class, true_label)
        ratio = (current_dist[sampled_class] + eps) / (ref_dist[sampled_class] + eps)
        group_log_probs.append(log_prob)
        group_rewards.append(reward)
        group_ratios.append(ratio)
    return group_log_probs, group_rewards, group_ratios

def compute_advantages(group_rewards, device):
    """
    Computes advantages as the z-score of the group rewards.
    Creates the tensor on the specified device.
    """
    eps = 1e-8
    rewards_tensor = torch.tensor(group_rewards, dtype=torch.float, device=device)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std() if rewards_tensor.std() > 0 else 1.0
    advantages = (rewards_tensor - mean_reward) / (std_reward + eps)
    return advantages

def compute_surrogate_loss(group_ratios, advantages, clip_eps):
    """
    Computes the surrogate loss for one example using PPO-style clipping.
    """
    ratios_tensor = torch.stack(group_ratios)  # (group_size,)
    clipped_ratios = torch.clamp(ratios_tensor, 1 - clip_eps, 1 + clip_eps)
    surrogate_terms = torch.min(ratios_tensor * advantages, clipped_ratios * advantages)
    return -surrogate_terms.mean()

def compute_kl_divergence(p_current, p_ref, eps=1e-8):
    """
    Computes the average KL divergence between current and reference distributions.
    """
    kl = torch.sum(p_current * (torch.log(p_current + eps) - torch.log(p_ref + eps)), dim=1)
    return kl.mean()
