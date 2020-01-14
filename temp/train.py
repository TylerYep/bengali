import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchsummary

import util
from util import Metrics, Mode
from args import init_pipeline
from dataset import load_train_data, INPUT_SHAPE
from models import ResNet34 as Model
from viz import visualize, compute_activations, compute_saliency
from tqdm import tqdm

def main():
    args, device = init_pipeline()
    criterion = nn.CrossEntropyLoss()
    model = Model().to(device)
    # torchsummary.summary(model, INPUT_SHAPE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = util.load_checkpoint(args.checkpoint, model, optimizer)
    run_name = checkpoint['run_name'] if checkpoint else util.get_run_name(args)
    start_epoch = checkpoint['epoch'] if checkpoint else 1

    def train_and_validate(loader, metrics, mode) -> float:
        if mode == Mode.TRAIN:
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        metrics.set_num_examples(len(loader))
        with tqdm(desc='', total=len(loader), ncols=120) as pbar:

            for i, (data, labels1, labels2, labels3) in enumerate(loader):
                data = data.to(device).unsqueeze(dim=1).float()
                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)

                if mode == Mode.TRAIN:
                    optimizer.zero_grad()

                if mode == Mode.TRAIN:
                    compute_activations(model, data, run_name)
                    # data.requires_grad_()

                outputs1, outputs2, outputs3 = model(data)
                loss1 = criterion(outputs1, labels1)
                loss2 = criterion(outputs2, labels2)
                loss3 = criterion(outputs3, labels3)

                total_loss = loss1.item() + loss2.item() + loss3.item()

                if mode == Mode.TRAIN:
                    (loss1 + loss2 + loss3).backward()
                    optimizer.step()
                    # compute_saliency(data, run_name)

                update_metrics(metrics, i, total_loss,
                                [outputs1, outputs2, outputs3],
                                [labels1, labels2, labels3], mode)
                pbar.update()
        return get_results(metrics, mode)

    metric_names = ('epoch_loss', 'running_loss', 'epoch_acc', 'running_acc')
    metrics = Metrics(SummaryWriter(run_name), metric_names, args.log_interval)
    train_loader = load_train_data(args)

    best_loss = np.inf
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch [{epoch}/{args.epochs}]')
        metrics.set_epoch(epoch)
        train_loss = train_and_validate(train_loader, metrics, Mode.TRAIN)
        # val_loss = train_and_validate(val_loader, metrics, Mode.VAL)

        # is_best = val_loss < best_loss
        # best_loss = min(val_loss, best_loss)
        # util.save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'rng_state': torch.get_rng_state(),
        #     'run_name': run_name,
        #     'epoch': epoch
        # }, run_name, is_best)


def update_metrics(metrics, i, loss, output, target, mode) -> None:
    outputs1, outputs2, outputs3 = output
    labels1, labels2, labels3 = target
    accuracy = (outputs1.argmax(1) == labels1).float().mean() \
             + (outputs2.argmax(1) == labels2).float().mean() \
             + (outputs3.argmax(1) == labels3).float().mean()
    metrics.update('epoch_loss', loss)
    metrics.update('running_loss', loss)
    metrics.update('epoch_acc', accuracy.item())
    metrics.update('running_acc', accuracy.item())

    if i % metrics.log_interval == 0:
        if i != 0: print(i)
        num_steps = (metrics.epoch-1) * metrics.num_examples + i
        metrics.write(f'{mode} Loss', metrics.running_loss / metrics.log_interval, num_steps)
        metrics.write(f'{mode} Accuracy', metrics.running_acc / metrics.log_interval, num_steps)
        metrics.reset(['running_loss', 'running_acc'])


def get_results(metrics, mode) -> float:
    total_loss = metrics.epoch_loss / metrics.num_examples
    total_acc = 100. * metrics.epoch_acc / (metrics.num_examples*3)
    print(f'{mode} Loss: {total_loss:.4f} Accuracy: {total_acc:.2f}%')
    metrics.write(f'{mode} Epoch Loss', total_loss, metrics.epoch)
    metrics.write(f'{mode} Epoch Accuracy', total_acc, metrics.epoch)
    metrics.reset()
    return total_loss


if __name__ == '__main__':
    main()
