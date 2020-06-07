import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset 

class ClassificationDataset(Dataset):

    def __init__(self, texts: list, labels: list, seq_len: int, target_type: str = 'flipped', num = None):


        self.labels = labels
        self.texts = texts
        self.seq_len = seq_len

        self.target_type = target_type
        self.num = num


    def pad_sequences(self, arr):

        if len(arr) > self.seq_len:

            arr = arr[:self.seq_len]

        arr = torch.tensor(arr)

        op = torch.zeros((self.seq_len))

        op[:arr.size(0)] = arr

        return op



    def flip(self, tnsr):

        tnsr = tnsr.unsqueeze(0)
        tnsr = torch.flip(tnsr, (0, 1))

        return tnsr.squeeze(0)

    
    def empty_zeros(self):

        return torch.zeros((self.seq_len))

    def empty_ones(self):

        return torch.ones((self.seq_len))

    
    def empty_like(self, number: int):

        op = self.empty_zeros()

        op[op == 0] = number

        return op


    def __len__(self):

        return len(self.texts)

    def __getitem__(self, ix):

        label = self.labels[ix]
        text = self.texts[ix]
        text = self.pad_sequences(text)

        if self.target_type == 'flipped':
            target = self.flip(text)
        elif self.target_type == 'empty_zeros':
            target = self.empty_zeros()
        elif self.target_type == 'empty_ones':
            target = self.empty_ones()
        elif self.target_type == 'empty_like':
            if not self.num:
                return f'Please provide a value in the `num` argument in the bounds of your vocab size'
            else:
                target = self.empty_like(self.num)

        return {"src": text.long(), "label": torch.tensor(label), "trg": target.long()}


