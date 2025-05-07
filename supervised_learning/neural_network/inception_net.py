import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# hyperparameters
data_path = '/data'
BATCH_SIZE = 32
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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        return F.relu(x, inplace=True)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        outinception_3x3_reduced,
        outinception_3x3,
        outinception_5x5_reduced,
        outinception_5x5,
        out_pool
    ):
        super().__init__()

        self.branch1 = ConvBlock(
            in_channels, out_1x1, kernel_size=1, stride=1
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, outinception_3x3_reduced, kernel_size=1),
            ConvBlock(outinception_3x3_reduced, outinception_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, outinception_5x5_reduced, kernel_size=1),
            ConvBlock(outinception_5x5_reduced, outinception_5x5, kernel_size=3, padding=1),
            ConvBlock(outinception_5x5, outinception_5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        return torch.cat([y1, y2, y3, y4], 1)


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.first_layers = nn.Sequential(
            ConvBlock(in_channel, 192, kernel_size=3, padding=1)
        )

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(1024, out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        out = self.first_layers(x)

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.max_pool(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.max_pool(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        return self.fc(out)



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

    model = Inception(in_channel=1, out_channel=12).to(device)
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
            torch.save(model.state_dict(), 'inception_net_best.pth')
            print(f"Epoch {epoch+1}: Best model saved (val_loss < train_loss)")
        else:
            torch.save(model.state_dict(), 'inception_net.pth')
            print(f"Epoch {epoch+1}: Normal checkpoint saved")

    accuracy = check_accuracy(train_loader, model)
    print(f"Accuracy on training set: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()


