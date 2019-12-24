import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
# from tqdm import tqdm

import util
# from util import AverageMeter
from dataset import load_data
from models import ResNet18
# from viz import visualize


def train_model(args, model, criterion, train_loader, optimizer, epoch, writer):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # summary(model, (1, 64, 64))

    running_acc, running_loss = 0.0, 0.0
    epoch_acc, epoch_loss = 0.0, 0.0
    for i, (data, labels1, labels2, labels3) in enumerate(train_loader):
        data = data.to(device).unsqueeze(1).float()
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)
        optimizer.zero_grad()

        # if args.visualize:
        #     visualize(data, labels1)

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

        epoch_loss += total_loss
        epoch_acc += output1_diff + output2_diff + output3_diff

        (loss1 + loss2 + loss3).backward()
        optimizer.step()


        if i % args.log_interval == 0 and i != 0:
            print(i)
            num_steps = (epoch-1) * len(train_loader) + i
            writer.add_scalar('training loss', running_loss / args.log_interval, num_steps)
            writer.add_scalar('training accuracy', running_acc / args.log_interval, num_steps)
            running_acc, running_loss = 0.0, 0.0

    print('train_acc : {:.2f}%'.format(100*epoch_acc/(len(train_loader)*3)))
    print('train_loss : {:.4f}'.format(epoch_loss/len(train_loader)))
    writer.add_scalar('training epoch loss', epoch_loss / len(train_loader), epoch)
    writer.add_scalar('training epoch accuracy', epoch_acc / (len(train_loader)*3), epoch)
    return 0


def validate_model(args, model, criterion, val_loader, epoch, writer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    running_acc, running_loss = 0.0, 0.0
    epoch_acc, epoch_loss = 0.0, 0.0
    with torch.no_grad():
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

            epoch_loss += total_loss
            epoch_acc += output1_diff + output2_diff + output3_diff

    print('val_acc : {:.2f}%'.format(100*epoch_acc/(len(val_loader)*3)))
    print('val_loss : {:.4f}'.format(epoch_loss/len(val_loader)))
    writer.add_scalar('val epoch loss', epoch_loss / len(val_loader), epoch)
    writer.add_scalar('val epoch accuracy', epoch_acc / (len(val_loader)*3), epoch)
    return epoch_loss


def main():
    args = util.get_args()
    util.set_seed()

    ###
    model = ResNet18()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    ###

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)
        start_epoch = checkpoint['epoch']

    run_name = util.get_run_name()
    writer = SummaryWriter(run_name)
    train_loader, val_loader = load_data(args)

    best_loss = np.inf
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.99)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        train_loss = train_model(args, model, criterion, train_loader, optimizer, epoch, writer)
        val_loss = validate_model(args, model, criterion, val_loader, epoch, writer)

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

        # scheduler.step()


if __name__ == '__main__':
    main()
