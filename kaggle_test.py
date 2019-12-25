import os
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

INPUT_SIZE = 64
DATA_PATH = '/kaggle/input/bengaliai-cv19'
CP_PATH = 'checkpoints/J'


def resize(df):
    resized = {}
    df = df.set_index('image_id')
    for i in tqdm(range(df.shape[0])):
        # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(INPUT_SIZE,INPUT_SIZE))
        image = cv2.resize(df.loc[df.index[i]].values,(INPUT_SIZE,INPUT_SIZE))
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T.reset_index()
    resized.columns = resized.columns.astype(str)
    resized.rename(columns={'index':'image_id'},inplace=True)
    return resized


class BengaliTestDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx][1:].values.reshape(64,64).astype(float)
        return image


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=2,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.block2 = nn.Sequential(
            nn.MaxPool2d(1,1),
            ResidualBlock(64,64),
            ResidualBlock(64,64,2)
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64,128),
            ResidualBlock(128,128,2)
        )

        self.block4 = nn.Sequential(
            ResidualBlock(128,256),
            ResidualBlock(256,256,2)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256,512),
            ResidualBlock(512,512,2)
        )

        self.avgpool = nn.AvgPool2d(2)
        # vowel_diacritic
        self.fc1 = nn.Linear(512,11)
        # grapheme_root
        self.fc2 = nn.Linear(512,168)
        # consonant_diacritic
        self.fc3 = nn.Linear(512,7)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1,x2,x3


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        super(ResidualBlock,self).__init__()
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self,x):
        residual = x
        x = self.cnn1(x)
        x = self.cnn2(x)
        x += self.shortcut(residual)
        x = nn.ReLU(True)(x)
        return x


def test_model(model, criterion):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = []
    for i in range(4):
        # data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/test_image_data_{i}.parquet')
        data = pd.read_feather(f'data/test_data_{i}.feather')

        data = resize(data)
        test_set = BengaliTestDataset(data)
        test_loader = DataLoader(test_set, batch_size=1000)

        with torch.no_grad():
            with tqdm(desc='Batch', total=len(test_loader), ncols=120, position=0, leave=True) as pbar:
                for i, data in enumerate(test_loader):
                    data = data.to(device)
                    outputs1, outputs2, outputs3 = model(data.unsqueeze(1).float())
                    predictions.append(outputs3.argmax(1).cpu().detach().numpy())
                    predictions.append(outputs2.argmax(1).cpu().detach().numpy())
                    predictions.append(outputs1.argmax(1).cpu().detach().numpy())
                    pbar.update()
    submission = pd.read_csv(f"data/sample_submission.csv")
    submission.target = np.hstack(predictions)
    submission.to_csv('submission.csv',index=False)


def main():
    set_seed()
    model = ResNet18()
    if CP_PATH != '':
        load_checkpoint(CP_PATH, model)

    criterion = nn.CrossEntropyLoss()
    test_model(model, criterion)


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_run, model, optimizer=None):
    """ Loads model parameters (state_dict) from file_path. If optimizer is provided,
    loads state_dict of optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint = torch.load(os.path.join(CP_PATH, 'model_best.pth.tar'))
    # map_location=torch.device('cpu'))
    torch.set_rng_state(checkpoint['rng_state'])
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


if __name__ == '__main__':
    main()