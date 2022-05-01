import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from model.CusFish import FishNet
from util.util_data import ColorAugmentation, CIFAR10, data_load
from util.util_metric import show_metric

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def main():
    model_config = config.net_config
    env_config = config.envs
    device = config.device

    model = FishNet(**model_config).to(device)
    model.load_state_dict(torch.load(f'save/model/model_Imp_4_E75.pt'))
    print(f"# of Parameter : {count_parameters(model)}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=env_config['lr'], momentum=env_config['momentum'],
                                weight_decay=env_config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = setup_data(env_config['batch_size'])

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    print(f"Iterations : {len(train_loader)}")
    for epoch in range(env_config['epochs']):
        train_loss = train(model, train_loader, device, criterion, optimizer)
        train_loss_per_epoch.append(train_loss)

        val_loss = validate(model, val_loader, device, criterion)
        val_loss_per_epoch.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            true, pred = test(model, test_loader, device)
            show_metric(true, pred)
            print(f"Epoch {epoch} / Train Loss: {train_loss} / Val Loss: {val_loss}")
            torch.save(model.state_dict(), f'save/model/model_{env_config["Implement_ID"]}_E{epoch}.pt')

    losses = {"Train": train_loss_per_epoch, "Val": val_loss_per_epoch}
    show_loss(losses, env_config['Implement_ID'])

    # true, pred = test(model, test_loader, device)
    # show_metric(true, pred)


def train(model, loader, device, criterion, optimizer):
    model.train()
    loss_per_iter = 0
    for i, (data, label) in tqdm(enumerate(loader)):
        if i > 155:
            continue
        data = data.to(device)
        label = label.long().to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_per_iter += loss.detach().cpu().numpy()
    return loss_per_iter / len(loader)


def validate(model, loader, device, criterion):
    model.eval()
    val_loss_per_iter = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.long().to(device)

            output = model(data)
            loss = criterion(output, label)
            val_loss_per_iter += loss.detach().cpu().numpy()

    return val_loss_per_iter / len(loader)


def test(model, loader, device):
    trues = []
    preds = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)

            trues.extend(label.detach().cpu().numpy())
            preds.extend(pred.detach().cpu().numpy())

    return trues, preds


def setup_data(batch_size):
    input_size = 224
    ratio = 224.0 / float(input_size)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorAugmentation(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(256 * ratio)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    x_train, x_val, y_train, y_val = data_load()
    x_test, y_test = data_load(purpose='Test')
    train_dataset = CIFAR10(x_train, y_train, transform=train_transform)
    val_dataset = CIFAR10(x_val, y_val, transform=train_transform)
    test_dataset = CIFAR10(x_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def show_image(sample):
    if type(sample) == torch.Tensor:
        sample = sample.detach().cpu().numpy().astype(np.uint8)

    if sample.shape[-1] != 3:
        sample = sample.T
    print()
    sample = Image.fromarray(sample)

    plt.figure()
    plt.imshow(sample)
    plt.show()


def show_loss(losses: dict, idx):
    pd.DataFrame(losses).to_csv(f'save/loss_log/loss_log_{idx}.csv')
    plt.figure()
    for label, loss in losses.items():
        plt.plot(loss, label=label)
    plt.legend()
    plt.savefig(f'save/loss_fig/loss_fig_{idx}.png')
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    main()
print()


