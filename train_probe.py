import time
import random
import torch
import argparse
from freeze_dried_data import RFDD
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------------------------------------------
# Core components (safe to import elsewhere)
# ------------------------------------------------------------------
class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, data_name, activation_file, activation_key, ids):
        """Generic dataset that returns (activation, label) for a given key."""
        self.activation_key = activation_key
        self.dataset = RFDD(f"data/{data_name}/{activation_file}")
        self.ids = ids if ids is not None else self.dataset.keys()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = self.dataset[self.ids[idx]]
        return row[self.activation_key].squeeze(0), row["label"]


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim: int = 4096, output_dim: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_epoch(model, loader, optim):
    """Return (avg_loss, accuracy) for one training epoch."""
    model.train()
    for acts, labels in loader:
        logits = model(acts).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()
        optim.step(); optim.zero_grad()

def eval_epoch(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for acts, labels in loader:
            logits = model(acts).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            preds = torch.round(torch.sigmoid(logits))
            correct += (preds == labels).float().sum()
            total += len(labels)
            loss_sum += loss.item()
    return correct / total, loss_sum / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Monkeys")
    parser.add_argument("--activation_file", default="prompt_activations.fdd", help="FDD file containing activations")
    parser.add_argument("--train_key", default="system_prompt_activations", help="Key used for training activations")
    parser.add_argument("--val_keys", default="ood_prompt_activations,red_team_prompt_activations", help="Comma-separated list of activation keys to validate on")
    parser.add_argument("--val_pct", default=0.20, type=float)
    parser.add_argument("--seed", default=6, type=int)
    parser.add_argument("--exp_name", default="", help="Name of the experiment")
    args = parser.parse_args()

    # Fix random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    readable_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = f"{args.data_name}_{args.exp_name}_{readable_time}".strip("_")

    # Output directory
    output_dir = os.path.join("outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Data split -----------------------------
    dataset = RFDD(f"data/{args.data_name}/{args.activation_file}")
    all_ids = list(range(len(dataset)))
    random.shuffle(all_ids)
    split_idx = int((1 - args.val_pct) * len(all_ids))
    train_ids, test_ids = all_ids[:split_idx], all_ids[split_idx:]

    train_dataset = ProbeDataset(args.data_name, args.activation_file, args.train_key, train_ids)
    test_dataset = ProbeDataset(args.data_name, args.activation_file, args.train_key, test_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Build validation loaders for each specified key
    val_keys = [k.strip() for k in args.val_keys.split(",") if k.strip()]
    if args.train_key not in val_keys:
        val_keys.append(args.train_key)  # ensure in-domain val set present
    val_loaders = {
        k: torch.utils.data.DataLoader(
            ProbeDataset(args.data_name, args.activation_file, k, test_ids),
            batch_size=64,
            shuffle=False,
        )
        for k in val_keys
    }
    val_loaders['training_set'] = torch.utils.data.DataLoader(
            ProbeDataset(args.data_name, args.activation_file, k, train_ids),
            batch_size=64,
            shuffle=False,
        )

    # ---------------------- Model & Optim ---------------------------
    model = LinearProbe(input_dim=4096, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # ---------------------- Training loop ---------------------------
    num_epochs = 100
    train_losses, train_accs = [], []
    val_acc_history = {k: [] for k in val_keys}
    val_losses_history = {k: [] for k in val_keys}

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer)
        val_accs, val_losses = {k: eval_epoch(model, loader) for k, loader in val_loaders.items()}

        # Log metrics
        for k in val_keys:
            val_acc_history[k].append(val_accs[k])
            val_losses_history[k].append(val_losses[k])

        val_acc_str = ", ".join([
            f"{k}: {val_accs[k]:.4f}" for k in val_keys
        ])
        print(
            f"Epoch {epoch}, Metrics:{val_acc_str}"
        )

    # ---------------------- Plot accuracy vs time -------------------
    plt.figure(figsize=(10, 6))
    for k in val_keys:
        plt.plot(range(num_epochs), val_acc_history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    # ---------------------- Plot loss vs time -----------------------
    plt.figure(figsize=(10, 6))
    for k in val_keys:
        plt.plot(range(num_epochs), val_losses_history[k], label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # ---------------- Final Evaluation & Plots ----------------------
    for key, loader in val_loaders.items():
        model.eval()
        pos_scores, neg_scores, correct, total = [], [], 0, 0
        with torch.no_grad():
            for acts, labels in loader:
                logits = model(acts).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = torch.round(probs)
                correct += (preds == labels).float().sum()
                total += len(labels)
                pos_scores.extend(probs[labels == 1].cpu().numpy())
                neg_scores.extend(probs[labels == 0].cpu().numpy())

        final_acc = correct / total
        print(f"Final {key} Acc: {final_acc:.4f}")

        plt.figure(figsize=(12, 8))
        sns.violinplot(data=[pos_scores, neg_scores], orient="h", cut=0)
        plt.yticks([0, 1], ["Positive", "Negative"])
        plt.xlabel("Probe Score")
        plt.xlim(0, 1)
        plt.title(f"Probe Scores for {args.data_name} ({key})")
        plt.savefig(os.path.join(output_dir, f"violin_{key}.png"))
        plt.close()

    # ---------------------- Save final model ------------------------
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))


