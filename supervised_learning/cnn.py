# classic imports
import torch
import os 
from PIL import Image
import torch.nn.functional as F  
import torchvision.transforms as transforms 
from torch import optim 
from torch import nn  
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm  
import pandas as pd


# hyperparameters
in_channels = 3
num_classes = 2
learning_rate = 3e-4 
batch_size = 1
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR_PATH = "dataset"
LABELS_DIR_PATH = "labels.csv"
BATCH_SIZE = 1
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])


# dataset
class LoadDataset(Dataset):
    def __init__(self, labels_dir_path, image_dir_path, transform=None):
        self.annotations = pd.read_csv(labels_dir_path)
        self.image_dir_path = image_dir_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir_path, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image,y_label
    

# model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjust the input size based on your image dimensions
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# train loop
def train(model, train_loader, optimizer, criterion ):
    loop = tqdm(train_loader, leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


def main():
    # model, optimizer, loss
    model = CNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # load data
    train_dataset = LoadDataset(
        labels_dir_path=LABELS_DIR_PATH,
        image_dir_path=IMAGE_DIR_PATH,
        transform=transformation,
    )
    num_samples = int(0.5 * len(train_dataset))
    train_subset, test_subset = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion)

    accuracy = check_accuracy(train_loader, model) 
    print(f"Accuracy on training set: {accuracy*100:.2f}")

    
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


if __name__ == "__main__":
    main()


   
    




