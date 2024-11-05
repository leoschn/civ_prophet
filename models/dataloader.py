import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

class RandomSwap:
    def __init__(self):
        pass

    def __call__(self, input):
        map_played, draft_1,draft_2, winner = input
        row_perm_1 = torch.randperm(draft_1.shape[0])
        row_perm_2 = torch.randperm(draft_2.shape[0])
        draft_1 = draft_1[row_perm_1]
        draft_2 = draft_2[row_perm_2]
        return (map_played, draft_1, draft_2, winner)

class Dataset_draft(Dataset):

    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.map_played = self.data['Map played']
        self.draft_W =  self.data['PickW1'], self.data['PickW2'], self.data['PickW3'], self.data['PickW4']
        self.draft_L =  self.data['PickL1'], self.data['PickL2'], self.data['PickL3'], self.data['PickL4']
        self.transform = RandomSwap()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # format : (map,draft1,draft2,winner)
        if bool(random.getrandbits(1)):
            input =  (torch.tensor(self.map_played[idx]), torch.tensor(
                [self.draft_W[0][idx], self.draft_W[1][idx], self.draft_W[2][idx], self.draft_W[3][idx]])
                    , torch.tensor(
                [self.draft_L[0][idx], self.draft_L[1][idx], self.draft_L[2][idx], self.draft_L[3][idx]]),
                    torch.tensor([1,0]))
        else :
            input = (torch.tensor(self.map_played[idx])
                    , torch.tensor(
                [self.draft_L[0][idx], self.draft_L[1][idx], self.draft_L[2][idx], self.draft_L[3][idx]])
                    , torch.tensor(
                    [self.draft_W[0][idx], self.draft_W[1][idx], self.draft_W[2][idx], self.draft_W[3][idx]]),
                    torch.tensor([0,1]))
        return self.transform(input)

def load_data(path,split,batch_size):
    dataset = Dataset_draft(path)
    [dataset_train, dataset_test] = torch.utils.data.random_split(dataset, split)
    return DataLoader(dataset_train,batch_size),DataLoader(dataset_test, batch_size)