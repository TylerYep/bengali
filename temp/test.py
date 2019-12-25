import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# if torch.cuda.is_available():
#     from tqdm import tqdm_notebook as tqdm
# else:
from tqdm import tqdm

import util
from dataset import load_test_data
from models import ResNet18


def test_model(args, model, criterion, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    predictions = []
    with torch.no_grad():
        with tqdm(desc='Batch', total=len(test_loader), ncols=120, position=0, leave=True) as pbar:
            for i, data in enumerate(test_loader):
                data = data.to(device)
                outputs1, outputs2, outputs3 = model(data.unsqueeze(1).float())
                predictions.append(outputs3.argmax(1).cpu().detach().numpy())
                predictions.append(outputs2.argmax(1).cpu().detach().numpy())
                predictions.append(outputs1.argmax(1).cpu().detach().numpy())
                pbar.update()
    submission = pd.read_csv("data/sample_submission.csv")
    submission.target = np.hstack(predictions)

    return 0


def main():
    args = util.get_args()
    util.set_seed()

    test_loader = load_test_data(args)
    model = ResNet18()
    if args.checkpoint != '':
        util.load_checkpoint(args.checkpoint, model)

    criterion = nn.CrossEntropyLoss()
    test_model(args, model, criterion, test_loader)


if __name__ == '__main__':
    main()
