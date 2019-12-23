import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
if torch.cuda.is_available():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import util
from util import AverageMeter
from dataset import load_data
from models import ResNet18
from test import test_model, validate_model
from viz import visualize


def train_model(args, model, criterion, train_loader, optimizer, epoch, writer):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # summary(model, (1, 64, 64))

    running_acc, running_loss = 0.0, 0.0
    acc, loss = 0.0, 0.0
    with tqdm(desc='Batch', total=len(train_loader), ncols=120, position=1, leave=True) as pbar:
        for i, (data, labels1, labels2, labels3) in enumerate(train_loader):
            data = data.to(device).unsqueeze(1).float()
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)
            optimizer.zero_grad()

            if args.visualize:
                visualize(data, labels1)

            outputs1, outputs2, outputs3 = model(data)
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

            (loss1 + loss2 + loss3).backward()
            optimizer.step()

            num_steps = (epoch-1) * len(train_loader) + i

            if i % args.log_interval == 0:
                writer.add_scalar('training loss', loss / args.log_interval, num_steps)
                writer.add_scalar('training accuracy', acc / args.log_interval, num_steps)
                acc, loss = 0.0, 0.0

            pbar.set_postfix({'Accuracy': f'{running_acc / (len(train_loader)*3):.5f}',
                              'Loss': running_loss / len(train_loader)})
            pbar.update()

    return 0


def main():
    args = util.get_args()
    util.set_seed()

    ###
    model = ResNet18()
    optimizer = optim.Adam(model.parameters(), lr=4e-4)
    criterion = nn.CrossEntropyLoss()
    ###

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = checkpoint['epoch']

    run_name = util.get_run_name()
    writer = SummaryWriter(run_name)
    train_loader, val_loader, _ = load_data(args)

    best_loss = np.inf
    with tqdm(desc='Epoch', total=args.epochs + 1 - start_epoch, ncols=120, position=0, leave=True) as pbar:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_model(args, model, criterion, train_loader, optimizer, epoch, writer)
            val_loss = validate_model(args, model, criterion, val_loader)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            print(f"Saving model at Epoch {epoch}")
            util.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'run_name': run_name,
                'epoch': epoch
            }, run_name, is_best)

            pbar.update()


if __name__ == '__main__':
    main()
