import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, optim, cuda, nn, stack, sigmoid
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import csv

from data import data_conversion as dc


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vectDim = 2
        self.embedding = nn.Embedding(4, self.vectDim)
        self.l1 = nn.Linear(53 * self.vectDim, 53)
        self.l2 = nn.Linear(53, 20)
        self.l3 = nn.Linear(20, 14)
    
    def forward(self, input):
        input = input.t()
        batch_size = input.size(0)
        embedded = self.embedding(input)
        embedded = embedded.view(batch_size, -1)
        o1 = F.relu(self.l1(embedded))
        o2 = F.relu(self.l2(o1))
        o3 = self.l3(o2)
        return o3


def target2tensor(target, batch_size):
    # vecs = torch.zeros((len))
    y_onehot = torch.FloatTensor(batch_size, 14)
    y_onehot.zero_()
    y_onehot.scatter_(1, target.view(-1, 1), 1)
    return y_onehot


def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(stack(data)), Variable(target.clone().detach())
        optimizer.zero_grad()
        output = model(data)
        # print(output.size(), target.size())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()



        if batch_idx % 50 == 0:
            # print([round(x * 100) for x in F.softmax(output[0], dim=-1).tolist()], target[0].item())
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))            

def test(csvwriter):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = Variable(stack(data)), Variable(target.clone().detach())
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        csvwriter.writerow([round(x, 4) for x in F.softmax(output[0], dim=-1).tolist()] + [target[0].item()])

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    with open('test_results.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',', quotechar='\"')
        for epoch in range(5):
            train()
            test(csvwriter)
        

