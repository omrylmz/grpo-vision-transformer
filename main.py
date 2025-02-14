import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy


# -----------------------------
# 1. Define a Vision Transformer for Binary Classification
# -----------------------------

class ViTClassifier(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, num_heads=4, num_layers=2, mlp_ratio=4, num_classes=2):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Class token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding (for all patches + class token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x):
        # x shape: (batch, in_channels, image_size, image_size)
        batch_size = x.shape[0]
        # Divide image into patches
        patches = x.unfold(2, x.shape[2] // (x.shape[2] // 4), x.shape[2] // (x.shape[2] // 4)) \
            .unfold(3, x.shape[3] // (x.shape[3] // 4), x.shape[3] // (x.shape[3] // 4))
        # For CIFAR10 (32x32) with patch_size=4, this yields shape (batch, in_channels, 8, 8, 4, 4)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)  # (B, num_patches, patch_dim)

        # Project patches to embeddings
        patch_embeddings = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, patch_embeddings), dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Permute for transformer: (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)  # (seq_len, batch, embed_dim)

        # Take the class token (first token)
        cls_out = x[0]
        logits = self.mlp_head(cls_out)  # (batch, num_classes)
        return logits


# -----------------------------
# 2. Data Preparation: Filter CIFAR10 for Cat (label 3) and Dog (label 5)
# -----------------------------

def get_cat_dog_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # Filter for cat (class 3) and dog (class 5)
    def filter_cat_dog(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
        subset = Subset(dataset, indices)
        return subset

    train_subset = filter_cat_dog(trainset)
    test_subset = filter_cat_dog(testset)

    # Remap labels: cat -> 0, dog -> 1
    def remap_labels(batch):
        # batch is a tuple (image, label) from the Subset, so we need to remap label.
        image, label = batch
        new_label = 0 if label == 3 else 1
        return image, new_label

    # Wrap the Subset with a simple collate function that remaps labels
    def collate_fn(batch):
        images, labels = zip(*[remap_labels(item) for item in batch])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader


# -----------------------------
# 3. Supervised Training Loop
# -----------------------------

def supervised_train(model, train_loader, test_loader, num_epochs=5, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Test Acc = {test_acc:.4f}")

    # Plot training loss and test accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Supervised Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy")

    plt.tight_layout()
    plt.show()
    return train_losses, test_accuracies


# -----------------------------
# 4. GRPO RL Training Loop
# -----------------------------
# Here we implement GRPO (Group Relative Policy Optimization) on top of our trained model.
# For each image in a batch, we will sample multiple predictions (from the model's softmax),
# compute a binary reward (1 if sampled prediction == ground truth, else 0),
# compute the probability ratios with respect to a reference model,
# clip the ratios, add a KL divergence penalty, and backpropagate.

def compute_kl_divergence(p_current, p_ref):
    # p_current, p_ref: tensors of shape (batch, num_classes)
    # We compute KL(p_current || p_ref) for each sample.
    # Use log probabilities of p_current and treat p_ref as target.
    # F.kl_div expects log_target as input? Actually, F.kl_div(input, target, reduction='batchmean') computes:
    # sum(target * (log(target) - input)). We want KL(p_current || p_ref) = sum p_current * (log p_current - log p_ref)
    # We can compute that manually.
    kl = torch.sum(p_current * (torch.log(p_current + 1e-8) - torch.log(p_ref + 1e-8)), dim=1)
    return kl.mean()


def grpo_rl_train(model, train_loader, num_rl_iters=3, group_size=5, clip_eps=0.2, beta=0.04, rl_lr=1e-4,
                  device='cuda'):
    """
    Run GRPO reinforcement learning on the model.

    For each example in a batch:
      - Compute current model distribution (p_current) over classes.
      - Using p_current, sample group_size predictions.
      - Compute binary reward: 1 if sampled label equals ground truth, else 0.
      - For each sample, compute ratio = p_current(sample) / p_ref(sample), where p_ref is computed from a reference model.
      - Compute advantage for the group as z-score: (reward - mean_reward) / (std_reward + eps).
      - Compute surrogate loss: L = - mean( min(ratio * advantage, clip(ratio, 1-clip_eps, 1+clip_eps) * advantage) )
      - Add KL divergence penalty between current and reference distribution.
      - Backpropagate and update the model.

    The reference model is taken as a frozen copy of the model at the beginning of each RL iteration.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rl_lr)

    rl_losses = []
    # For a number of RL iterations
    for rl_iter in range(num_rl_iters):
        model.train()
        running_loss = 0.0
        count = 0

        # Create a reference model (frozen copy)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)
                # Get current logits and probabilities for the batch
                logits = model(images)  # shape: (batch, num_classes)
                p_current = F.softmax(logits, dim=1)  # shape: (batch, num_classes)
                # Get reference model probabilities
                ref_logits = ref_model(images)  # shape: (batch, num_classes)
                p_ref = F.softmax(ref_logits, dim=1)  # shape: (batch, num_classes)

                # For each example in the batch, sample group_size predictions from current distribution.
                # We will collect for each example:
                # - log probability of the sampled prediction (from current model)
                # - reward (1 if equals true label, else 0)
                # - probability ratio: p_current(sample) / (p_ref(sample) + eps)
                sample_log_probs = []
                sample_rewards = []
                sample_ratios = []

                for i in range(batch_size):
                    # Get the distribution for the i-th example: shape (num_classes,)
                    dist = p_current[i]
                    ref_dist = p_ref[i]
                    true_label = labels[i].item()

                    group_log_probs = []
                    group_rewards = []
                    group_ratios = []

                    for _ in range(group_size):
                        # Sample an action from the current distribution.
                        sampled_class = torch.multinomial(dist, num_samples=1).item()
                        # Get log probability from current model (log-softmax)
                        log_prob = torch.log(dist[sampled_class] + 1e-8)
                        # Reward: 1 if correct, else 0.
                        reward = 1.0 if sampled_class == true_label else 0.0
                        # Ratio: current probability / reference probability
                        ratio = (dist[sampled_class] + 1e-8) / (ref_dist[sampled_class] + 1e-8)

                        group_log_probs.append(log_prob)
                        group_rewards.append(reward)
                        group_ratios.append(ratio)

                    # Convert group lists to tensors.
                    group_log_probs = torch.stack(group_log_probs)  # (group_size,)
                    group_rewards = torch.tensor(group_rewards, device=device, dtype=torch.float)
                    group_ratios = torch.tensor(group_ratios, device=device, dtype=torch.float)

                    # Compute advantage as z-score.
                    mean_reward = group_rewards.mean()
                    std_reward = group_rewards.std() if group_rewards.std() > 0 else 1.0
                    advantages = (group_rewards - mean_reward) / (std_reward + 1e-8)

                    # Compute surrogate loss for this example:
                    # For each sample, compute:
                    #   surrogate = min(ratio * advantage, clip(ratio, 1-clip_eps, 1+clip_eps) * advantage)
                    clipped_ratio = torch.clamp(group_ratios, 1 - clip_eps, 1 + clip_eps)
                    surrogate = torch.min(group_ratios * advantages, clipped_ratio * advantages)
                    # We want to maximize the surrogate, so the loss is negative.
                    example_loss = - surrogate.mean()

                    sample_log_probs.append(example_loss)
                    sample_rewards.append(advantages)  # not used directly in loss here
                    sample_ratios.append(group_ratios)  # not used directly

                # Average loss over batch
                batch_loss = torch.stack(sample_log_probs).mean()

                # Compute KL divergence between current and reference distributions for the batch.
                kl_loss = compute_kl_divergence(p_current, p_ref)

                total_loss = batch_loss + beta * kl_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item() * batch_size
                count += batch_size

        avg_rl_loss = running_loss / count
        rl_losses.append(avg_rl_loss)
        print(f"RL Iteration {rl_iter + 1}/{num_rl_iters}, RL Loss: {avg_rl_loss:.4f}")

    # Visualize RL loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_rl_iters + 1), rl_losses, marker='o')
    plt.xlabel("RL Iteration")
    plt.ylabel("Average RL Loss")
    plt.title("GRPO RL Loss Over Iterations")
    plt.show()


# -----------------------------
# 5. Main: Supervised Training then GRPO RL Fine-Tuning
# -----------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    num_supervised_epochs = 5
    supervised_lr = 1e-3

    # Get DataLoaders (cat vs. dog)
    train_loader, test_loader = get_cat_dog_dataloaders(batch_size=batch_size)

    # Instantiate the model
    model = ViTClassifier(image_size=32, patch_size=4, in_channels=3,
                          embed_dim=64, num_heads=4, num_layers=2, num_classes=2)

    print("Starting supervised training...")
    sup_train_losses, sup_test_acc = supervised_train(model, train_loader, test_loader,
                                                      num_epochs=num_supervised_epochs,
                                                      lr=supervised_lr, device=device)

    # Evaluate supervised performance
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    initial_acc = correct / total
    print(f"Supervised model accuracy: {initial_acc:.4f}")

    # Now perform GRPO RL fine-tuning
    print("Starting GRPO RL fine-tuning...")
    grpo_rl_train(model, train_loader, num_rl_iters=3, group_size=5, clip_eps=0.2,
                  beta=0.04, rl_lr=1e-4, device=device)

    # Evaluate RL-fine-tuned performance
    model.eval()
    correct = 0
    total = 0
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
