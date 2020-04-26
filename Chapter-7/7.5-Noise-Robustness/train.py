  
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def update_function(param, grad, loss, learning_rate):
      return param - learning_rate * grad


def train(args, model, noisy_model, device, train_loader, epoch):
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    noisy_losses = AverageMeter('Noisy Loss', ':.4f')

    # Learning rate schedule
    learning_rate = args.gamma ** (epoch-1) * args.lr

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Create noisy model
        mp = list(model.parameters())
        mcp = list(noisy_model.parameters())
        for i in range(0, len(mp)):
            mcp[i].data[:] = mp[i].data[:]
        with torch.no_grad():
            for param in noisy_model.parameters():
                param.add_(torch.randn(param.size()).to(device) * args.train_eta)

        # Zero out the gradients
        noisy_model.zero_grad()
        # Compute output, loss and backward pass for the noisy model
        noisy_output = noisy_model(data)
        noisy_loss = F.nll_loss(noisy_output, target)
        noisy_loss.backward()
        noisy_losses.update(noisy_loss.item(), data.size(0))
        with torch.no_grad():
            output = model(data)
            loss = F.nll_loss(output, target)
            losses.update(loss.item(), data.size(0))

        # Use gradients from noisy model to update the parameters of the original model
        with torch.no_grad():
            for name, p in model.named_parameters():
                for noisy_name, noisy_p in noisy_model.named_parameters():
                    if noisy_name == name:
                        new_val = update_function(p, noisy_p.grad, noisy_loss, learning_rate)
                        p.copy_(new_val)
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t{} - {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), noisy_losses, losses))


def test(args, model, noisy_model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTrained Model - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    noisy_model.eval()
    test_loss = 0
    correct = 0
    # Create noisy model
    mp = list(model.parameters())
    mcp = list(noisy_model.parameters())
    for i in range(0, len(mp)):
        mcp[i].data[:] = mp[i].data[:]
    with torch.no_grad():
        for param in noisy_model.parameters():
            param.add_(torch.randn(param.size()).to(device) * args.test_eta)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = noisy_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    noisy_accuracy = 100. * correct / len(test_loader.dataset)
    print('Noisy Model   - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), noisy_accuracy))

    return accuracy, noisy_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--train-eta', type=float, default=0.03,
                        help="STD of random perturbation to add to the weights at training")
    parser.add_argument('--test-eta', type=float, default=0.03,
                        help="STD of random perturbation to add to the weights at test time")

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/tmp/data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    model_noisy = Net().to(device)

    for epoch in range(1, args.epochs + 1):
        train(args, model, model_noisy, device, train_loader, epoch)
        accuracy, noisy_accuracy = test(args, model, model_noisy, device, test_loader)

    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"Final noisy accuracy: {noisy_accuracy:.2f}%")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()