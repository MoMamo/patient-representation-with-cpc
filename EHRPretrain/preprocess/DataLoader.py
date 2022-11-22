from torch.utils.data import Dataset, DataLoader
import torch
from preprocess import transform
from torchvision import transforms
import pandas as pd
import numpy as np


class EHR2VecDset(Dataset):
    def __init__(self, dataset, params):
        # dataframe preproecssing
        # filter out the patient with number of visits less than min_visit
        self.data = dataset
        self._compose = transforms.Compose([
            transform.TruncateSeqence(params['max_seq_length']),
            transform.CreateSegandPosition(),
            transform.TokenAgeSegPosition2idx(params['token_dict_path'], params['age_dict_path']),
            transform.RetriveSeqLengthAndPadding(params['max_seq_length']),
            transform.FormatAttentionMask(params['max_seq_length']),
        ])

    def __getitem__(self, index):
        """
        return: age, code, position, segmentation, mask, label
        """
        patid = self.data.patid[index]

        s = np.random.choice(len(self.data.code[index]), 1)[0]

        sample = {
            'code': self.data.code[index][s:],
            'age': self.data.age[index][s:],
            'event': self.data.event[index],
            'time': self.data.time[index],
            # 'label': self.data.label[index]
        }

        sample = self._compose(sample)

        return {'patid': torch.LongTensor([int(patid)]),
                'code': torch.LongTensor(sample['code']),
                'age': torch.LongTensor(sample['age']),
                'att_mask': torch.LongTensor(sample['att_mask']),
                'event': torch.FloatTensor([sample['event']]),
                'time': torch.FloatTensor([sample['time']]),
                'len': torch.LongTensor([sample['len']])
                }

    def __len__(self):
        return len(self.data)


def CPCDataLoader(params, data, batch_size, shuffle=True):
    print('data size:', len(data))
    dset = EHR2VecDset(dataset=data, params=params)
    dataloader = DataLoader(dataset=dset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=params['num_workers'],
                            )
    return dataloader
