from typing import Tuple
from argparse import Namespace
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
if torch.cuda.is_available():
    DATA_PATH = '../gdrive/My Drive/Colab Notebooks/data'
else:
    DATA_PATH = 'data'

INPUT_SHAPE = (1, 64, 64)
INPUT_SIZE = 64


def load_train_data(args):
    # norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = BengaliDataset(f"{DATA_PATH}/train-orig.csv")
    val_set = BengaliDataset(f"{DATA_PATH}/test.csv")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    return train_loader, val_loader


def load_test_data(args):
    test_set = BengaliTestDataset(f"{DATA_PATH}/test.csv")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return test_loader


class BengaliDataset(Dataset):
    """ Dataset for training a model on a dataset. """

    def __init__(self, data_path, transform=None):
        super().__init__()
        # col_names = ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']
        label = pd.read_csv(data_path)
        data0 = pd.read_feather(f'{DATA_PATH}/train_data_0.feather')
        # data1 = pd.read_feather(f'{DATA_PATH}/train_data_1.feather')
        # data2 = pd.read_feather(f'{DATA_PATH}/train_data_2.feather')
        # data3 = pd.read_feather(f'{DATA_PATH}/train_data_3.feather')
        data_full = pd.concat([data0], ignore_index=True)

        reduced_index = label.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']) \
                             .apply(lambda x: x.sample(5)).image_id.values
        reduced_train = label.loc[label.image_id.isin(reduced_index)]
        reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

        self.data = reduced_data
        self.label = reduced_train
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx][1:].values.reshape(INPUT_SIZE, INPUT_SIZE).astype(np.float)

        # if self.transform:
        #     transformed = self.transform(image=img)
        #     img = transformed['image']

        label1 = self.label.vowel_diacritic.values[idx]
        label2 = self.label.grapheme_root.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        return image, label1, label2, label3


class BengaliTestDataset(Dataset):
    """ Dataset for training a model on a dataset. """

    def __init__(self, data_path, transform=None):
        super().__init__()

        test = pd.read_csv(data_path)
        data0 = pd.read_feather(f'{DATA_PATH}/test_data_0.feather')
        data1 = pd.read_feather(f'{DATA_PATH}/test_data_1.feather')
        data2 = pd.read_feather(f'{DATA_PATH}/test_data_2.feather')
        data3 = pd.read_feather(f'{DATA_PATH}/test_data_3.feather')
        data_full = pd.concat([data0,data1,data2,data3], ignore_index=True)
        self.data = data_full
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx][1:].values.reshape(INPUT_SIZE, INPUT_SIZE).astype(np.float)

        # if self.transform:
        #     transformed = self.transform(image=img)
        #     img = transformed['image']

        return image
