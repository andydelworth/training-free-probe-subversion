import random
import torch
import argparse
from freeze_dried_data import RFDD

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Monkeys')
parser.add_argument('--val_pct', default=.20, type=float)
args = parser.parse_args()

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, data_name, is_train=True, ids=None):
        self.is_train = is_train
        self.ids = ids
        # this file stores list of (text, activations, label)
        self.dataset = RFDD(f"data/{data_name}/activations.fdd")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.dataset[self.ids[idx]]['activations'].squeeze(0), self.dataset[self.ids[idx]]['label']


class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

'''
partition data keys into train and validation sets
'''
dataset = RFDD(f"data/{args.data_name}/activations.fdd")
all_ids = list(range(len(dataset)))
random.shuffle(all_ids)
train_ids = all_ids[:int((1-args.val_pct)*len(all_ids))]
test_ids = all_ids[int((1-args.val_pct)*len(all_ids)):]

'''
create dataloaders
'''
train_dataset = ProbeDataset(args.data_name, is_train=True, ids=train_ids)
test_dataset = ProbeDataset(args.data_name, is_train=False, ids=test_ids)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)

'''
create model
'''
model = LinearProbe(input_dim=4096, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

'''
training loop
'''
def train_epoch(model, train_loader, optimizer):
    train_correct, train_total = 0, 0
    for activations, labels in train_loader:
        pred_logits = model(activations).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred_labels = torch.round(torch.sigmoid(pred_logits))
        train_correct += (pred_labels == labels).float().sum()
        train_total += len(labels)
    return train_correct / train_total

def test_epoch(model, test_loader):
    correct, total = 0, 0
    pos_scores, neg_scores = [], []
    for activations, labels in test_loader:
        pred_logits = model(activations).squeeze(1)
        pred_probs = torch.sigmoid(pred_logits)
        pred_labels = torch.round(pred_probs)
        correct += (pred_labels == labels).float().sum()
        total += len(labels)
        
        # Collect scores for positive and negative examples
        pos_scores.extend(pred_probs[labels == 1].detach().numpy())
        neg_scores.extend(pred_probs[labels == 0].detach().numpy())
    
    # Create violin plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    plot_data = [pos_scores, neg_scores]
    sns.violinplot(data=plot_data)
    plt.xticks([0, 1], ['Positive', 'Negative'])
    plt.ylabel('Probe Score')
    plt.title('Distribution of Probe Scores by Label')
    plt.savefig('probe_score_dist.png')
    plt.close()
    
    return correct / total

for epoch in range(10):
    train_acc = train_epoch(model, train_loader, optimizer)
    test_acc = test_epoch(model, test_loader)
    print(f"Epoch {epoch}, Train Acc: {train_acc:.4f}, Eval Acc: {test_acc:.4f}")


