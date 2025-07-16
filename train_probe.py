import torch
import argparse

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, is_train=True):
        self.dataset_folder = dataset_folder
        self.is_train = is_train
        # this file stores list of (text, activations, label)
        self.dataset = torch.load(f"{dataset_folder}/{'train' if is_train else 'test'}.pt")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][1], self.dataset[idx][2]


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_folder", type=str, required=True)
args = parser.parse_args()

dataset_folder = args.dataset_folder

