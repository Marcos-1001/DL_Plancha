import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64, 3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 28 * 28, 128) 
        self.fc2 = nn.Linear(128, 6)

    
    
    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        print(x.shape)

        x = self.fc1(x)
        print(x.shape)
        #x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, loss_fn, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset)}]")

def test(model, test_loader, device): 
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            test_loss += nn.functional.cross_entropy(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
