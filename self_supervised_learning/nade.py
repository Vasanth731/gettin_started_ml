# the NADE model is taken from simonjisu and it's modified for inference 

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt

# few needed functions
def draw_sampling(model):
    model.eval()
    x_hat, xs, nll_loss = model.sample(n=1, only_prob=True)
    fig, axes = plt.subplots(1, 2)
    for x, ax in zip([x_hat, xs], axes):
        ax.matshow(x.cpu().detach().squeeze().view(28, 28).numpy())
        ax.axis('off')
    plt.show()
    
def non_decreasing(L):
    """for early stopping"""
    return all(x<=y for x, y in zip(L, L[1:]))

def load_model(model_path="nade-binary.pt", input_dim=784, hidden_dim=500, device='cpu'):
    model = NADE(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def sample_from_model(model, n_samples=5, device='cpu'):
    with torch.no_grad():
        x_hat, xs, nll_loss = model.sample(n=n_samples, only_prob=True)
    return x_hat.cpu(), xs.cpu(), nll_loss.cpu()

def plot_samples(samples, probs, n=5):
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
    for i in range(n):
        axes[i, 0].imshow(probs[i].view(28, 28), cmap='gray')
        axes[i, 0].set_title("Probability")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(samples[i].view(28, 28), cmap='gray')
        axes[i, 1].set_title("Sample")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


# model
class NADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NADE, self).__init__()
        self.D = input_dim
        self.H = hidden_dim
        self.params = nn.ParameterDict({
            "V" : nn.Parameter(torch.randn(self.D, self.H)),
            "b" : nn.Parameter(torch.zeros(self.D)),
            "W" : nn.Parameter(torch.randn(self.H, self.D)),
            "c" : nn.Parameter(torch.zeros(1, self.H)),
        })
        nn.init.xavier_normal_(self.params["V"])
        nn.init.xavier_normal_(self.params["W"])
        
    def forward(self, x):
        # a0: (B, H)
        a_d = self.params["c"].expand(x.size(0), -1)
        # Compute p(x)
        x_hat = self._cal_prob(a_d, x)
        
        return x_hat
    
    def _cal_prob(self, a_d, x, sample=False):
        """
        assert 'x = None' when sampling
        Parameters:
         - a_d : (B, H)
         - x : (B, D)
         
        Return:
         - x_hat: (B, D), estimated probability dist. of batch data
        """
        if sample:
            assert (x is None), "No input for sampling as first time"
        
        x_hat = []  # (B, 1) x D
        xs = []
        for d in range(self.D):
            # h_d: (B, H)
            h_d = torch.sigmoid(a_d)
            # p_hat: (B, H) x (H, 1) + (B, 1) = (B, 1)
            p_hat = torch.sigmoid(h_d.mm(self.params["V"][d:d+1, :].t() + self.params["b"][d:d+1]))
            x_hat.append(p_hat)
            
            if sample:
                # random sample x: (B, 1) > a_{d+1}: (B, 1) x (1, H)
                x = torch.distributions.Bernoulli(probs=p_hat).sample()
                xs.append(x)
                a_d = x.mm(self.params["W"][:, d:d+1].t()) + self.params["c"]
                
            else:
                # a_{d+1}: (B, 1) x (1, H)
                a_d = x[:, d:d+1].mm(self.params["W"][:, d:d+1].t()) + self.params["c"]
        
        # x_hat: (B, D), estimated probability dist. of batch data
        x_hat = torch.cat(x_hat, 1)
        if sample:
            xs = torch.cat(xs, 1)
            return x_hat, xs
        return x_hat
    
    def _cal_nll(self, x_hat, x):
        nll_loss = -1 * ( x*torch.log(x_hat + 1e-10) + (1-x)*torch.log(1-x_hat + 1e-10))
        return nll_loss
    
    def sample(self, n=1, only_prob=True):
        a_d = self.params["c"].expand(n, -1)  # (n, H)
        # Compute p(x)
        x_hat, xs = self._cal_prob(a_d, x=None, sample=True)
        nll_loss = self._cal_nll(x_hat, xs)
        return (x_hat, xs, nll_loss)


def train_one_epoch(train_loader, loss_function, optimizer, model, device):
    model.train()
    total_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        
        optimizer.zero_grad()
        # preprocess to binary
        inputs = imgs.view(imgs.size(0), -1).gt(0.).float().to(device)
        x_hat = model(inputs)
        loss = loss_function(x_hat, inputs)
        loss.backward()
        optimizer.step()
        
        # record
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"\t[{i*imgs.size(0)/len(train_loader.dataset)*100:.2f}%] loss: {loss/imgs.size(0):.4f}")
            
    return total_loss


def test_one_epoch(test_loader, loss_function, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, _ in test_loader:
            # preprocess to binary
            inputs = imgs.view(imgs.size(0), -1).gt(0.).float().to(device)
            x_hat = model(inputs)
            loss = loss_function(x_hat, inputs)
            total_loss += loss.item()
            
        print(f"\t[Test Result] loss: {total_loss/len(test_loader.dataset):.4f}")
    return total_loss/len(test_loader.dataset)
        

def main(draw=False):
    data_path = Path(".").absolute().parent / "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_step = 10
    batch_size = 512

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root=str(data_path), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.MNIST(root=str(data_path), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = NADE(input_dim=784, hidden_dim=500).to(device)
    loss_function = nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
    
    train_losses = []
    test_losses = []
    best_loss = 99999999
    wait = 0
    for step in range(n_step):
        print(f"Running Step: [{step+1}/{n_step}]")
        train_loss = train_one_epoch(train_loader, loss_function, optimizer, model, device)
        test_loss = test_one_epoch(test_loader, loss_function, model, device)
        scheduler.step()
        # sampling
        if draw:
            draw_sampling(model)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss <= best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "nade-binary2.pt")
            print(f"\t[Model Saved]")
            if (step >= 2) and (wait <= 3) and (non_decreasing(test_losses[-3:])):
                wait += 1
            elif wait > 3:
                print(f"[Early Stopped]")
                break
            else:
                continue


def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    n_samples = 5

    print("Sampling from NADE model...")
    probs, samples, nll_loss = sample_from_model(model, n_samples=n_samples, device=device)
    print(f"NLL loss for generated samples: {nll_loss.mean().item():.4f}")

    plot_samples(samples, probs, n=n_samples)
                
                
if __name__ == "__main__":

    main(draw=True)
    infer()