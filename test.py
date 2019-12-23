import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import util
from dataset import load_data
from models import ResNet18


def validate_model(args, model, criterion, val_loader, epoch, writer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    acc, loss = 0.0, 0.0
    running_acc, running_loss = 0.0, 0.0
    with torch.no_grad():
        with tqdm(desc='Batch', total=len(val_loader), ncols=120, position=1, leave=True) as pbar:
            for i, (data, labels1, labels2, labels3) in enumerate(val_loader):
                data = data.to(device)
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)

                outputs1, outputs2, outputs3 = model(data.unsqueeze(1).float())
                loss1 = criterion(outputs1, labels1)
                loss2 = criterion(outputs2, labels2)
                loss3 = criterion(outputs3, labels3)

                total_loss = loss1.item() + loss2.item() + loss3.item()
                output1_diff = (outputs1.argmax(1) == labels1).float().mean()
                output2_diff = (outputs2.argmax(1) == labels2).float().mean()
                output3_diff = (outputs3.argmax(1) == labels3).float().mean()

                running_loss += total_loss
                running_acc += output1_diff
                running_acc += output2_diff
                running_acc += output3_diff

                loss += total_loss
                acc += output1_diff
                acc += output2_diff
                acc += output3_diff

                num_steps = (epoch-1) * len(val_loader) + i

                if i % args.log_interval == 0:
                    writer.add_scalar('val loss', loss / args.log_interval, num_steps)
                    writer.add_scalar('val accuracy', acc / args.log_interval, num_steps)
                    acc, loss = 0.0, 0.0

                pbar.set_postfix({'Accuracy': f'{running_acc / (len(val_loader)*3):.5f}',
                                'Loss': running_loss / len(val_loader)})
                pbar.update()

    return running_loss


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

    _, _, test_loader = load_data(args)
    model = ResNet18()
    if args.checkpoint != '':
        util.load_checkpoint(args.checkpoint, model)

    criterion = F.nll_loss
    test_model(args, model, criterion, test_loader)


if __name__ == '__main__':
    main()
