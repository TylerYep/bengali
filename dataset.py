import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import pandas as pd
if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

INPUT_SIZE = 64

class_map = pd.read_csv("data/class_map.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")
test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")


def load_data(args):
    # norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = BengaliDataset("data/train.csv")
    val_set = BengaliDataset("data/dev.csv", val=True)
    test_set = BengaliTestDataset("data/test.csv")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)
    return train_loader, val_loader, test_loader


class BengaliDataset(Dataset):
    """ Dataset for training a model on a dataset. """

    def __init__(self, data_path, val=False, transform=None):
        super().__init__()
        col_names = ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme']
        train = pd.read_csv(data_path)
        data_full = pd.read_feather('data/train_data_0.feather')
        # data1 = pd.read_feather('data/train_data_1.feather')
        # data2 = pd.read_feather('data/train_data_2.feather')
        # data3 = pd.read_feather('data/train_data_3.feather')
        # data_full = pd.concat([data0,data1,data2,data3], ignore_index=True)

        reduced_index = train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']) \
                             .apply(lambda x: x.sample(5)).image_id.values
        reduced_train = train.loc[train.image_id.isin(reduced_index)]
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
        data_full = pd.read_feather('data/test_data_0.feather')
        # data1 = pd.read_feather('data/test_data_1.feather')
        # data2 = pd.read_feather('data/test_data_2.feather')
        # data3 = pd.read_feather('data/test_data_3.feather')
        # data_full = pd.concat([data0,data1,data2,data3], ignore_index=True)
        self.data = data_full
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx][1:].values.reshape(100,100).astype(np.float)

        # if self.transform:
        #     transformed = self.transform(image=img)
        #     img = transformed['image']

        return image
