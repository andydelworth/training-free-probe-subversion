import random
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default='Monkeys')
args = parser.parse_args()

class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.is_train = is_train
        # this file stores list of (text, activations, label)
        self.dataset = torch.load(f"{dataset_folder}/{'train' if is_train else 'test'}.pt")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][1], self.dataset[idx][2]


def train_test_split(data, pct=.2):
    random.shuffle(data)
    # train, test
    return data[:int((1-pct)*len(data))], data[int((1-pct)*len(data)):]

dataset_folder = args.dataset_folder

with open(os.path.join(f'data/{dataset_folder}/data.json'), 'r') as f:
    data = json.load(f)



