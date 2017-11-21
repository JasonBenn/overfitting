import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Adapted from PyTorch MNIST example')
parser.add_argument('--experiment', type=int, required=True, help='experiment #, determines folder to store weights and results')

args = parser.parse_args()
EXPERIMENT = 'experiment_' + str(args.experiment)
os.mkdir(EXPERIMENT)

BATCH_SIZE = 64
EPOCHS = 2
LOG_INTERVAL = 100
LR = .01
MOMENTUM = 0.5
NO_CUDA = True
SEED = 1
TEST_BATCH_SIZE = 1000

CUDA = not NO_CUDA and torch.cuda.is_available()

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)


kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if CUDA:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

def train(epoch):
    model.train()
    loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(list(model.parameters()), EXPERIMENT + "/train_{}.params".format(epoch))
    return loss.data[0]  # POSSIBLE BUG: is this the right value to return?

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


train_losses = []
test_losses = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train(epoch)
    train_losses.append((epoch, train_loss))
    test_loss = test()
    test_losses.append((epoch, test_loss))

torch.save(train_losses, EXPERIMENT + '/train_losses')
torch.save(test_losses, EXPERIMENT + '/test_losses')
