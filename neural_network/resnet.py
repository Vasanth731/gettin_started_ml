# residual_neural_network 
# weights saved/used - resnet_best_classifier_checkpoint.pth
# weights saved/used - resnet_best_classifier_aroundwhat_checkpoint.pth

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime 

# hyperparameters
data_dir = '/home/htic/Pictures/FPUS23_Dataset/Dataset/four_poses/'
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_step_size = 16
start_epoch = 0
num_epochs = 50

# wandb login 
import wandb
wandb.login()
user = "vasanth-ambrose"
project = "resnet_classifier"
display_name = "visual"
wandb.init(entity=user, project=project, name=display_name)

# few needed functions
def train_log(loss,step_size):    
    print(f' loss {loss} step {step_size}')
    wandb.log({"Loss": loss},step=step_size)

def save_classifier_checkpoint(state,epoch,model_name):
    date=datetime.date(datetime.now())
    time=datetime.time(datetime.now())
    date_time=str(date)+str("__")+str(time)
    date_time=date_time[0:20]

    filename = f'/home/endodl/codes_vasanth/oct/ckpt/resnet_{model_name}.pth'

    print("=> Saving checkpoint")
    torch.save(state, filename)


# model
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channel=3, num_classes=39):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


# train 
def train_one_epoch(model,train_loader,optimizer,criterion,currentepoch):

    model = model
    optimizer = optimizer

    model.train()
    correct = 0
    total = 0
    
    loop = tqdm(train_loader)

    for batch_idx, (images,labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx%log_step_size==0:
            print(f'EPOCH : {currentepoch}')
            print(f'loss  : {loss}')

            # Plot(loss,(batch_idx*log_step_size)) 
            if currentepoch==0:
                train_log(loss.item(),(batch_idx*log_step_size))
            else:
                train_log(loss.item(),((batch_idx*log_step_size)+(len(train_loader)*log_step_size*currentepoch)))
            
        loop.set_postfix(loss=loss.item())

    # epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{currentepoch+1}/{num_epochs}] - Loss: {loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

    return model, optimizer

def val_one_epoch(model,validation_loader,criterion,currentepoch):

    model = model
    model.eval()
    val_correct = 0
    val_total = 0

    loop = tqdm(validation_loader)
    losses = []
    with torch.no_grad():
        for batch_idx, (val_images,val_labels) in enumerate(loop):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            
            loss = criterion(val_outputs,val_labels) 
            _, val_predicted = val_outputs.max(1)
            
            val_total += val_labels.size(0)
            val_correct += val_predicted.eq(val_labels).sum().item()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        val_accuracy = val_correct / val_total
        print(f"Epoch [{currentepoch+1}/{num_epochs}] - Val Loss: {loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
        mean_loss = np.mean(losses)

        return mean_loss


def train():

    data_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transform)
    num_classes = len(train_dataset.classes)

    # validation_ratio = 0.9
    # num_train = len(train_dataset)
    # num_validation = int(validation_ratio * num_train)
    # num_train = num_train - num_validation
    # train_subset, validation_subset = random_split(train_dataset, [num_train, num_validation])

    num_samples = int(0.5 * len(train_dataset))
    train_subset, validation_subset = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples])

    train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet50(img_channel=3, num_classes=39)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)


    best_val_loss = 1e9
    for currentepoch in range(start_epoch, num_epochs):
        wandb.watch(model,log="all")
        model, optimizer = train_one_epoch(model,train_loader,optimizer,criterion,currentepoch)
        mean_loss = val_one_epoch(model,validation_loader,criterion,currentepoch)

        dino_classifier_checkpoint = {
                "state_dict": model.state_dict(),
#                 "optimizer":optimizer.state_dict(),
            }
        model_name="classifier_checkpoint"
        save_classifier_checkpoint(dino_classifier_checkpoint,currentepoch,model_name)

        if mean_loss < best_val_loss:
            best_val_loss = mean_loss

            dino_classifier_checkpoint = {
                "state_dict": model.state_dict(),
#                 "optimizer":optimizer.state_dict(),
            }
            model_name="best_classifier_checkpoint"

            save_classifier_checkpoint(dino_classifier_checkpoint,currentepoch,model_name)

    return mean_loss

if __name__ == "__main__":
    train()