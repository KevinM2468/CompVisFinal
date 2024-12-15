import time
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_wildfire_dir = 'data/train/damage/wildfire'
train_hurricane_dir = 'data/train/damage/hurricane'
train_earthquake_dir = 'data/train/damage/earthquake'
train_no_damage_dir_root = 'data/train/no_damage'
val_wildfire_dir = 'data/validate/damage/wildfire'
val_hurricane_dir = 'data/validate/damage/hurricane'
val_earthquake_dir = 'data/validate/damage/earthquake'
val_no_damage_dir = 'data/validate/no_damage'
test_wildfire_dir = 'data/test/damage/wildfire'
test_hurricane_dir = 'data/test/damage/hurricane'
test_earthquake_dir = 'data/test/damage/earthquake'
test_no_damage_dir = 'data/test/no_damage'

no_damage_subs = ['earthquake', 'hurricane', 'wildfire']

image_height = 100
image_width = 100

categories = ['wildfire', 'hurricane', 'earthquake', 'no_damage']
num_classes = len(categories)

# data loading functions
def load_images(directory, label):
    """
    Loads images from a given directory, resizes, and appends them with the label.
    """
    import os
    ret_data = []
    
    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_height, image_width))
            ret_data.append([img, label])
        except Exception as e:
            continue
    
    return ret_data

def loadXY_Data(wildfireDir, hurricaneDir, earthquakeDir, noDamageDirRoot):
    """
    Loads the dataset from the given directories and returns X, Y as numpy arrays.
    """
    img_data = []
    print("\tLoading wildfire images...")
    img_data.extend(load_images(wildfireDir, 0))
    print("\tLoading hurricane images...")
    img_data.extend(load_images(hurricaneDir, 1))
    print("\tLoading earthquake images...")
    img_data.extend(load_images(earthquakeDir, 2))
    print("\tLoading no_damage images...")
    for i in range(3):
        dir_ = os.path.join(noDamageDirRoot, no_damage_subs[i])
        img_data.extend(load_images(dir_, 3))
    
    # shuffle the combined data
    print("\tShuffling data...")
    random.shuffle(img_data)
    
    # convert to numpy arrays
    print("\tConverting data to numpy arrays...")
    x_list = []
    y_list = []
    
    for features, label in img_data:
        x_list.append(features)
        y_list.append(label)
    
    x_array = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.int64)
    
    # normalize the data
    print("\tNormalizing data...")
    x_array /= 255.0
    
    return x_array, y_array

# load the data
print("Loading training data...")
x_train, y_train = loadXY_Data(train_wildfire_dir, train_hurricane_dir, train_earthquake_dir, train_no_damage_dir_root)
print("Creating validation set...")
x_val, y_val = loadXY_Data(val_wildfire_dir, val_hurricane_dir, val_earthquake_dir, val_no_damage_dir)
print("Creating test set...")
x_test, y_test = loadXY_Data(test_wildfire_dir, test_hurricane_dir, test_earthquake_dir, test_no_damage_dir)

print("Completed loading training, validation, and test data")

# make pytorch Datasets
class DisasterDataset(Dataset):
    def __init__(self, x_data, y_data):
        # x_data: (N, image_height, image_width, 3) in HWC format
        # y_data: (N,) labels
        self.x_data = torch.from_numpy(x_data).permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.y_data = torch.from_numpy(y_data)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

train_dataset = DisasterDataset(x_train, y_train)
val_dataset   = DisasterDataset(x_val,   y_val)
test_dataset  = DisasterDataset(x_test,  y_test)


# make pytorch dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


# define CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNModel, self).__init__()
        
        # 1) Conv2D(256, kernel_size=(3,3), padding='same', relu) -> AvgPool(2,2)
        # 2) Conv2D(256, (3,3), same, relu)
        # 3) Conv2D(256, (3,3), same, relu) -> MaxPool(2,2)
        # 4) Conv2D(128, (3,3), same, relu)
        # 5) Conv2D(128, (3,3), same, relu) -> MaxPool(2,2)
        # Flatten -> Dense(3500, relu) -> Dropout(0.5)
        # Dense(2000, relu) -> Dropout(0.2)
        # Dense(4, softmax)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # We need to figure out the dimensions at the Flatten layer.
        # After the above layers, the original 100x100 input is downsampled:
        #   - conv1 (same size) -> avgpool1 (down to 50x50)
        #   - conv2, conv3 (still 50x50) -> maxpool1 (down to 25x25)
        #   - conv4, conv5 (still 25x25) -> maxpool2 (down to 12x12 if it perfectly divides)
        # Actually 25/2 = 12 (integer floor), so final feature map is 128 x 12 x 12 = 128 * 12 * 12 = 18432
        
        self.flatten_dim = 128 * 12 * 12
        
        self.fc1 = nn.Linear(self.flatten_dim, 3500)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(3500, 2000)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(2000, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool2(x)
        
        x = x.view(-1, self.flatten_dim)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        logits = self.fc3(x)  # raw output (logits)
        return logits

# instantiate model, loss, optimizer
model = CNNModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training Loop
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)         # shape (batch_size, num_classes)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        # compute accuracy
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# main training process
num_epochs = 8
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f},   Val Acc: {val_acc:.4f}")


# plot results (Accuracy & Loss)
plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_acc_history, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_acc_history, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend(loc='best')
plt.title('Training vs Validation Accuracy')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_loss_history, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend(loc='best')
plt.title('Training vs Validation Loss')
plt.show()

# evaluate on test set
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# save the model
timestampString = 'model' + str(int(time.time()))
torch.save(model.state_dict(), timestampString + '.pth')
print("Model saved as", timestampString + ".pth")
