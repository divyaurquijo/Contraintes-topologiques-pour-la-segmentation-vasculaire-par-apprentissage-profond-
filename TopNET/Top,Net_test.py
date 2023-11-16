import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Define the TOPNet class
class TOPNet(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(TOPNet, self).__init__()
      
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.MaxPool2d(3, 1), 
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(64, 128, 3, 1, 1)
        )
        
        # Decoder : Vessel extraction 
        self.decoder1 = nn.Sequential(
            nn.Softmax(),
            nn.Conv2d(1,1,1,2),
            nn.Conv2d(1,8,1,2),
            nn.ConvTranspose2d(), 
        )

        # Decoder : Centerness
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Sigmoid for output in [0, 1]
        )

        # Decoder : Topological distance decoder
        self.decoder3 = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Sigmoid for output in [0, 1]
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        
        # Split into mean and logvar
        mu, logvar = torch.chunk(x, 2, dim=-1)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar

# Instantiate the TOPNet
input_size = 784  # for example, if you're working with MNIST images
hidden_size = 256
latent_size = 32

vae = TOPNet(input_size, hidden_size, latent_size)

# Define the loss function
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

# Define your optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_data in your_data_loader:
        # Flatten input if needed
        x = batch_data.view(-1, input_size)

        # Forward pass
        x_recon, mu, logvar = vae(x)

        # Calculate loss
        loss = vae_loss(x_recon, x, mu, logvar)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the reconstruction loss at each epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
