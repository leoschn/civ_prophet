from modulefinder import Module

import pandas as pd
import torch
import torch.nn as nn

class win_predictor(torch.nn.Module):

    def __init__(self,drop_rate):
        super().__init__()
        self.ohe_civ = nn.Embedding(78, 2)
        self.ohe_map = nn.Embedding(8, 2)

        self.predictor = nn.Sequential(nn.Linear(8*2+2,8),
                                       nn.Dropout(p=drop_rate),
                                       nn.Linear(8,4),
                                       nn.Dropout(p=drop_rate),
                                       nn.Linear(4,2))

    def forward(self, map, draft1, draft2):
        draft = torch.concat([draft1,draft2],dim=-1)
        draft_ohe = self.ohe_civ(draft)
        map_ohe = self.ohe_map(map)
        input = torch.concat([draft_ohe.flatten(),map_ohe],dim=-1)
        pred = self.predictor(input)
        return torch.sigmoid(pred)

    def forward_embedding(self, civ):
        return self.ohe_civ(civ)
