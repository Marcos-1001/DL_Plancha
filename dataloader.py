# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
from random import randint


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        
        # input shape: 3x32x32


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels= 50, out_channels=40, kernel_size=3) 
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=30, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(30*28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, 40)
        self.fc4 = nn.Linear(64, 30)




    
    
    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


        
        return F.log_softmax(x, dim=1)

class CNN_NEW(nn.Module):
    def __init__(self):
        super(CNN_NEW, self).__init__()

        # Input shape: 3x32x32

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=50, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=50, out_channels=40, kernel_size=3) 
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=30, kernel_size=3)

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Calculate the output size after convolution and pooling layers
        # Input size (32x32) -> Conv1 -> (30x30) -> Pool -> (15x15)
        # Conv2 -> (13x13) -> Pool -> (6x6)
        # Conv3 -> (4x4)

        self.fc1 = nn.Linear(30 * 4 * 4, 128)  # Adjusted size based on convolution output
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 40)
        self.fc4 = nn.Linear(40, 30)

    def forward(self, x):
        # Convolutional layers with activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # Flatten the tensor
        x = x.view(-1, 30 * 4 * 4)

        # Fully connected layers with activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for final output

        return x

def train(model, train_loader, optimizer, loss_fn,  device):
    model.train()
    final_loss = 0
    accuracy = 0
    correct = 0 

    for batch, (X, y) in tqdm(enumerate(train_loader)):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
        final_loss += loss.item()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset)}]")
    

    final_loss = loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f"Train Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {final_loss:>8f} \n")

    return final_loss, accuracy
        

def test(model, test_loader, device): 
    model.eval()
    test_loss, correct = 0, 0
    #  Confussion matrix

    all_preds = []
    all_targets = []


    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            all_preds.extend(y_hat.argmax(1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

            test_loss += nn.functional.cross_entropy(y_hat, y).item()
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    




    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(confusion_matrix(all_targets, all_preds))

    
    


    return test_loss, correct

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
    def __init__(self, root, images_root, transform=None, augmented_data = False):
        self.root_dir = root
        train_csvs = ["train_1", "train_2", "train_3"]
        
        
        self.data = []

        for csv in train_csvs:
            with open(f"{root}/{csv}.csv") as f:
                for line in f:
                    img_paths = []
                    img_path, label = line.strip().split(',')
                    if img_path == 'ImageId':
                        continue
                    elif not img_path.endswith('.png'):
                        img_path = f"{img_path}.png"
                    
                    if augmented_data:
                        img_paths = [f"{images_root}/{csv}/augmented/{img_path[:-4]}_augmented_{randint(0,7)}.png" for i in range(randint(0, 4))]
                    else:
                        img_path =  f"{images_root}/{csv}/{img_path}"
                        img_paths = [img_path]
                    #print(img_path, int(label))
                    for i in range(len(img_paths)):
                        self.data.append((img_paths[i], int(label)))
                         
        
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





transform = transforms.Compose([ 
                                    transforms.Resize((32, 32)), 
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(10),
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
                                    # gaussian noise
                                    transforms.GaussianBlur(3),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                    transforms.ToTensor()]
                        )
# Mixing datasets
train_dataset = CustomDataset("/teamspace/studios/this_studio",'/teamspace/studios/this_studio/NIVEL1/NIVEL1/TRAIN', transform=transform, augmented_data=False)

# partition the dataset into train and test sets
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
print(len(train_dataset), len(test_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

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

# RESNET
import torchvision.models as models

model_name = 'CNN_NEW'

if model_name == 'resnet18':
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 30)
elif model_name == 'CNN_NEW':
    model = CNN_NEW()
else:
    model = CNN()



model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
epochs = 5

train_plot_loss = []
train_plot_accuracy = []

test_plot_loss = []
test_plot_accuracy = []

 # %%
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    tloss, taccuracy = train(model, train_loader, optimizer, loss_fn, device)
    testloss, testaccuracy =  test(model, test_loader, device)

    train_plot_loss.append(tloss)
    train_plot_accuracy.append(taccuracy)

    test_plot_loss.append(testloss)
    test_plot_accuracy.append(testaccuracy)
    save_model = f"/teamspace/studios/this_studio/DL_Plancha/models/{model_name}_2_epoch_{t+1}.pth"
    torch.save(model.state_dict(), save_model)


# %%




# %% 
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot([i.cpu().detach().numpy() for i in train_plot_loss ], label='train loss')
plt.plot(test_plot_loss , label='test loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_plot_accuracy, label='train accuracy')
plt.plot(test_plot_accuracy, label='test accuracy')
plt.title('Accuracy')
plt.legend()

# %%
# Visualize filters of the first convolutional layer


# Get the weights of the first convolutional layer
"""weights = model.conv1.weight.data
w = weights.cpu().numpy()


# Plot the weights
plt.figure(figsize=(16, 8))
for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(w[i].transpose(1, 2, 0))
    plt.axis('off')
    plt.title(f'Filter {i}')

"""

# %% 
#model = torch.load("/teamspace/studios/this_studio/DL_Plancha/models/CNN_NEW_1_epoch_1.pth")
test_folders = ["/teamspace/studios/this_studio/NIVEL1/NIVEL1/TEST1_3", "/teamspace/studios/this_studio/NIVEL2/NIVEL2/TEST2_3", "/teamspace/studios/this_studio/NIVEL3/NIVEL3/TEST3_3"]
#test_folder = "/teamspace/studios/this_studio/NIVEL2/NIVEL2/TEST2_3"
subfolders = ["newTest1", "newTest2", "newTest3"]
#make a csv file with the predictions
import csv
for test_folder in range(len(test_folders)):
    with open(f'submission_{test_folder}.csv', mode='w') as submission:
        submission_writer = csv.writer(submission, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        submission_writer.writerow(["ImageId", "Label"])
        for subfolder in subfolders:
            for img_ in glob(f"{test_folders[test_folder]}/{subfolder}/*.png"):

                img = Image.open(img_)
                img = img.convert('RGB')
                img = transform(img).unsqueeze(0)
                img = img.to(device)
                y_hat = model(img)
                y_hat = y_hat.argmax(1).item()
                submission_writer.writerow([img_[len(f"{test_folders[test_folder]}/{subfolder}/"):], y_hat])




