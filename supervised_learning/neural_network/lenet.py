import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import optim

# hyperparameters
data_path = '/home/htic/amb/data'
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
learning_rate = 3e-4

# utils
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
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# model
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        return self.c1(img)


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        return self.c2(img)


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        return self.c3(img)


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        return self.f4(img)


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        return self.f5(img)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5()

    def forward(self, img):
        output = self.c1(img)
        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output


def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, leave=False)

    for data, targets in loop:
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def val(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, leave=False):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)


def main():
    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    val_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = val(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lenet_best.pth')
            print(f"Epoch {epoch+1}: Best model saved (val_loss < train_loss)")
        else:
            torch.save(model.state_dict(), 'lenet.pth')
            print(f"Epoch {epoch+1}: Normal checkpoint saved")

    accuracy = check_accuracy(train_loader, model)
    print(f"Accuracy on training set: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
