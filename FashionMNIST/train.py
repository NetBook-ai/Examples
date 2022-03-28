import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
import model
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='resnet18', help="model")
parser.add_argument("--tbatch", type=int, default=64, help="batch size")
parser.add_argument("--nepochs", type=int, default=5, help="number of epochs")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on {}'.format(device))

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.tbatch, shuffle=True)

valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(valset, batch_size=args.tbatch, shuffle=True)
print('Training on FashionMNIST')


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0

    # Set mode
    if(train):
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        with(torch.set_grad_enabled(train)):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        if(train):
        	#backpropagate
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += torch.sum(pred == y.detach())
    return epoch_loss / len(loader), epoch_acc.double() / len(loader.dataset)


if __name__ == '__main__':

    net = model.__dict__[args.model]().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = run_model(net, train_loader, criterion, optimizer)
        val_loss, val_acc = run_model(net, val_loader, criterion, optimizer, False)
        end = time.time()

        stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t
                time: {:.1f}s""".format(e+1, train_loss, train_acc, val_loss, val_acc, end - start)
        print(stats)