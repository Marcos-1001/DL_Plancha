# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        
        # input shape: 3x50x50

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels= 50, out_channels=40, kernel_size=3) 
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(40*46*46, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)




    
    
    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


        
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, loss_fn,  device):
    model.train()
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

# %%
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import numpy as np

from glob import glob

 
class CustomDataset(Dataset):
    def __init__(self, root, transform=None, ):
        self.root_dir = root
        # Label based on the folder
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}
        
        self.data = []
        for i, c in enumerate(self.classes):
            for file in glob(f'{root}/{c}/*.jpg'):
                self.data.append((file, i))       
                
                         
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def data_augmentation(self, img):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Random rotation
        angle = np.random.randint(-10, 10)
        img = img.rotate(angle)
        return img


def get_data_loader(root, batch_size):
    transform = transforms.Compose([ 
                                    transforms.Resize((50, 50)), 
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor()]
                                )    
    dataset = CustomDataset(root, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


train_loader = get_data_loader('data/seg_train/seg_train', 64)
test_loader = get_data_loader('data/seg_test/seg_test', 1000)
# Plot the first batch of images
# %%
import matplotlib.pyplot as plt

for X, y in train_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    plt.figure(figsize=(16, 16))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        plt.imshow(X[i].numpy().T)
        plt.title(f"{y[i].item()}")
        plt.axis('off')        
    break

# %% 
model  = CNN() 
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 10
# %%
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_loader, optimizer, loss_fn, device)
    test(model, test_loader, device)
# %%
# Visualize filters of the first convolutional layer

# Get the weights of the first convolutional layer
weights = model.conv1.weight.data
w = weights.cpu().numpy()


# Plot the weights
plt.figure(figsize=(16, 8))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(w[i].transpose(1, 2, 0))
    plt.axis('off')
    plt.title(f'Filter {i}')




