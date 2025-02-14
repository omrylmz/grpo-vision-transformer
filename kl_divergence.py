import torch


# -----------------------------------------------------------------------------
# 4. Helper Function: KL Divergence Computation
# -----------------------------------------------------------------------------

def compute_kl_divergence(p_current, p_ref):
    # p_current, p_ref: tensors of shape (batch, num_classes)
    # We compute KL(p_current || p_ref) for each sample.
    # Use log probabilities of p_current and treat p_ref as target.
    # F.kl_div expects log_target as input? Actually, F.kl_div(input, target, reduction='batchmean') computes:
    # sum(target * (log(target) - input)). We want KL(p_current || p_ref) = sum p_current * (log p_current - log p_ref)
    # We can compute that manually.
    kl = torch.sum(p_current * (torch.log(p_current + 1e-8) - torch.log(p_ref + 1e-8)), dim=1)
    return kl.mean()
