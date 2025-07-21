import random
import torch
import argparse
from freeze_dried_data import RFDD
import matplotlib.pyplot as plt
import seaborn as sns

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
    model.train()
    correct, total = 0, 0
    for acts, labels in loader:
        logits = model(acts).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        loss.backward()
        optim.step(); optim.zero_grad()
        preds = torch.round(torch.sigmoid(logits))
        correct += (preds == labels).float().sum()
        total += len(labels)
    return correct / total

def eval_epoch(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for acts, labels in loader:
            logits = model(acts).squeeze(1)
            preds = torch.round(torch.sigmoid(logits))
            correct += (preds == labels).float().sum()
            total += len(labels)
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Monkeys")
    parser.add_argument("--activation_file", default="prompt_activations.fdd", help="FDD file containing activations")
    parser.add_argument("--train_key", default="system_prompt_activations", help="Key used for training activations")
    parser.add_argument("--val_keys", default="ood_prompt_activations,red_team_prompt_activations", help="Comma-separated list of activation keys to validate on")
    parser.add_argument("--val_pct", default=0.20, type=float)
    parser.add_argument("--seed", default=5, type=int)
    args = parser.parse_args()

    # Fix random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------- Data split -----------------------------
    dataset = RFDD(f"data/{args.data_name}/{args.activation_file}")
    all_ids = list(range(len(dataset)))
    random.shuffle(all_ids)
    split_idx = int((1 - args.val_pct) * len(all_ids))
    train_ids, test_ids = all_ids[:split_idx], all_ids[split_idx:]

    train_dataset = ProbeDataset(args.data_name, args.activation_file, args.train_key, train_ids)
    test_dataset = ProbeDataset(args.data_name, args.activation_file, args.train_key, test_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Build validation loaders for each specified key
    val_keys = [k.strip() for k in args.val_keys.split(",")]
    val_keys.append(args.train_key) # include the 'in-domain validation set
    val_loaders = {
        k: torch.utils.data.DataLoader(
            ProbeDataset(args.data_name, args.activation_file, k, test_ids),
            batch_size=2,
            shuffle=False,
        )
        for k in val_keys
    }

    # ---------------------- Model & Optim ---------------------------
    model = LinearProbe(input_dim=4096, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # ---------------------- Training loop ---------------------------
    for epoch in range(10):
        train_acc = train_epoch(model, train_loader, optimizer)
        val_accs = {k: eval_epoch(model, loader) for k, loader in val_loaders.items()}
        val_acc_str = ", ".join([
            f"{k}: {v:.4f}" for k, v in val_accs.items()
        ])
        print(f"Epoch {epoch}, Train Acc: {train_acc:.4f}, Val Accs -> {val_acc_str}")

    # ---------------- Final Evaluation & Checkpoint ----------------
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
        plt.savefig(f"probe_score_dist_{args.data_name}_{key}.png")
        plt.close()

    # Save checkpoint
    torch.save(model.state_dict(), f"{args.data_name}_linear_probe.pt")


