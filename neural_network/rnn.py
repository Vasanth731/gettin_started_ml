# classic imports
import torch
import torch.nn as nn 
from tqdm import tqdm
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28 # size of the input features at each time step
sequence_length = 28 # total number of words(time-steps) in the sentence(time_sequence)
num_layers = 2 # number of RNN layers stacked parallel to each other.
hidden_size= 256 # number of hidden nodes in the rnn
num_classes = 10
learning_rate = 0.001
batch_size = 4
num_epochs = 5


# Recurrent Neural Network model, change it to GRU, if you want to use Gated Recurrent Unit 
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # shape of x (batch_size * seq_len * input_size)
        out,_ = self.rnn(x, h0) # shape (batch_size * seq_len * hidden_size)
        out = out.reshape(out.shape[0], -1) # (64,28*256) flatten the layer
        out = self.fc(out) # shape (batch_size * num_classes)
        
        return out


# Long Short-Term Memory Networks 
# uncomment if you only want to use the last hidden layer alone in LSTM instead of using all hidden layers
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        # self.fc = nn.Linear(hidden_size , num_classes) # for takin only last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1) # comment this if u uncomment the below line 
        # out = self.fc(out[:,-1,:]) # for takin only last hidden state , shape (N,last hidden state,features)
        out = self.fc(out)

        return out
    
# Bidirectional LSTM
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    

def main():
    model = RNN(input_size,hidden_size,num_layers,num_classes).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = datasets.MNIST(root='./mnist_dataset/', train=True, transform=transforms.ToTensor(), download=False)
    num_samples = int(0.2 * len(train_dataset))
    train_subset,test_subset = random_split(train_dataset, [num_samples, len(train_dataset) - num_samples])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion)

    accuracy = check_accuracy(train_loader, model) 
    print(f"Accuracy on training set: {accuracy*100:.2f}")


def train(model, train_loader, optimizer, criterion):
    loop = tqdm(train_loader, leave=True)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device).squeeze(1) # to get a shape (64,28,28) from (64,1,28,28)
        targets = targets.to(device=device)
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


if __name__ == "__main__":
    main()