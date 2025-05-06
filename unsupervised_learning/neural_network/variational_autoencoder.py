# classic imports
import torch
from torch import nn
import torchvision.datasets as datasets  
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from tqdm import trange 

# hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 5
BATCH_SIZE = 4
LR_RATE = 3e-4 
SAVE_PATH = "generated_images" 

# model
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20): # h_dim = hidden_dimension | z_dim = latent_space dimension
        super().__init__()
        
        # encoder = q_phi(z|x) from input x it learns and outputs a latent_space z
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder = p_theata(x|z) from the latent_space z it outputs the reconstructed text, resembling input x
        # KL divergence will push the latent_space to learn and try to be a standard gaussian distribution 
        self.z_2hid = nn.Linear(z_dim, h_dim) 
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma*epsilon # this will make sure that it is gaussian , it's reparametrized
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mu, sigma # reconstructed_loss, mean, standard_deviation

# train loop
def train(train_loader, model, optimizer, loss_fn):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

def main():
    # model, optimizer and loss
    model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # loader
    dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
    num_samples = int(0.5 * len(dataset))
    train_subset, test_subset = random_split(dataset, [num_samples, len(dataset) - num_samples])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in trange(NUM_EPOCHS, desc="Training Epochs"):
        train(train_loader, model, optimizer, loss_fn)

    torch.save(model.state_dict(), "vae_weights.pt")

    return train_subset,test_subset, model

def inference(model,dataset,digit, num_examples=1):
    model = model.to("cpu")
    model.load_state_dict(torch.load("vae_weights.pt", map_location="cpu"))
    model.eval()

    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"{SAVE_PATH}/generated_{digit}_ex{example}.png")


if __name__ == "__main__":

    train_subset, test_subset, model = main()
    for idx in range(10):
        inference(model, test_subset, idx, num_examples=5)


# ANALYSIS
# 92 MB of GPU RAM (GTX 1650 Ti) 15W during training
# 1.4 MB of CPU RAM (AMD Ryzen 7 4800H)
# dataset downloaded form server and fetched from HDD
# 22 sec for 1 epoch | 340 iter/sec