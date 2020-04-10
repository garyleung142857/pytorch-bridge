import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, optim, cuda, nn, stack, sigmoid
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from data import data_conversion as dc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vectDim = 3
        self.embedding = nn.Embedding(4, self.vectDim)
        self.l1 = nn.Linear(53 * self.vectDim, 20)
        self.l2 = nn.Linear(20, 5)
        self.l3 = nn.Linear(5, 1)
    
    def forward(self, input):
        input = input.t()
        batch_size = input.size(0)
        embedded = self.embedding(input)
        embedded = embedded.view(batch_size, -1)
        o1 = F.relu(self.l1(embedded))
        o2 = F.relu(self.l2(o1))
        o3 = self.l3(o2)
        out = o3.view(-1)
        return out


def batch2tensor(batch):
    seq_tensor = torch.zeros()


def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data) 
        data, target = Variable(stack(data)), Variable(target.clone().detach().requires_grad_(True))
        # print(data, target)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))            


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # print(target)
        data, target = Variable(stack(data)), Variable(target.clone().detach().requires_grad_(True))
        output = model(data)
        # print(output)
        test_loss += criterion(output, target).item()
        correct += torch.lt(torch.abs(output - target), 0.5).sum().item()

    test_loss /= len(test_loader.dataset) / BATCH_SIZE
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')

if __name__ == "__main__":
    BATCH_SIZE = 256
    train_dataset = dc.DealParDataset(is_train_set=True)
    test_dataset = dc.DealParDataset(is_train_set=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    model = Net()
    device = 'cuda' if cuda.is_available() else 'cpu'
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(5):
        train()
        test()
        

