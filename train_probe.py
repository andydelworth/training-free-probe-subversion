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
    def __init__(self, data_name, is_train=True, ids=None):
        self.is_train = is_train
        self.ids = ids
        self.dataset = RFDD(f"data/{data_name}/activations.fdd")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.dataset[self.ids[idx]]["activations"].squeeze(0),
            self.dataset[self.ids[idx]]["label"],
        )


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim: int = 4096, output_dim: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# ------------------------------------------------------------------
# Everything below runs ONLY when this file is executed directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Monkeys")
    parser.add_argument("--val_pct", default=0.20, type=float)
    parser.add_argument("--seed", default=5, type=int)
    args = parser.parse_args()

    # Fix random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------------- Data split -----------------------------
    dataset = RFDD(f"data/{args.data_name}/activations.fdd")
    all_ids = list(range(len(dataset)))
    random.shuffle(all_ids)
    split_idx = int((1 - args.val_pct) * len(all_ids))
    train_ids, test_ids = all_ids[:split_idx], all_ids[split_idx:]

    train_dataset = ProbeDataset(args.data_name, is_train=True, ids=train_ids)
    test_dataset = ProbeDataset(args.data_name, is_train=False, ids=test_ids)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)

    # ---------------------- Model & Optim ---------------------------
    model = LinearProbe(input_dim=4096, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------- Training utils -------------------------
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

    # ---------------------- Training loop ---------------------------
    for epoch in range(10):
        train_acc = train_epoch(model, train_loader, optimizer)
        test_acc = eval_epoch(model, test_loader)
        print(f"Epoch {epoch}, Train Acc: {train_acc:.4f}, Eval Acc: {test_acc:.4f}")

    # ---------------- Final Evaluation & Checkpoint ----------------
    model.eval()
    pos_scores, neg_scores, correct, total = [], [], 0, 0
    with torch.no_grad():
        for acts, labels in test_loader:
            logits = model(acts).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = torch.round(probs)
            correct += (preds == labels).float().sum()
            total += len(labels)
            pos_scores.extend(probs[labels == 1].cpu().numpy())
            neg_scores.extend(probs[labels == 0].cpu().numpy())

    final_acc = correct / total
    print(f"Final Eval Acc: {final_acc:.4f}")

    plt.figure(figsize=(12, 8))
    sns.violinplot(data=[pos_scores, neg_scores], orient="h", cut=0)
    plt.yticks([0, 1], ["Positive", "Negative"])
    plt.xlabel("Probe Score")
    plt.xlim(0, 1)
    plt.title(f"Distribution of Probe Scores for {args.data_name}")
    plt.savefig(f"probe_score_dist_{args.data_name}.png")
    plt.close()

    # Save checkpoint
    torch.save(model.state_dict(), f"{args.data_name}_linear_probe.pt")


