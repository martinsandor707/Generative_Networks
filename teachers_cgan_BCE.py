# Import the most important utilities
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision.transforms as T
import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import patoolib
from tqdm import tqdm
import torch.nn.functional as F 

patoolib.extract_archive("train_cifar.zip",outdir="data_cifar")



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_paths = []
        self.labels = []

        # This will store what each number means.
        # For example:
        # 0 - buildings
        # 1 - forest
        # etc.
        self.label_dict = dict()

        # We go over each directory in the root directory (each directory represents a class)
        classes = [directory for directory in os.listdir(self.root_dir) if os.path.isdir(f"{self.root_dir}/{directory}")]
        for idx, _class in enumerate(classes):
            self.label_dict[idx] = _class

            # Then read the paths of the images one by one and store them in our lists
            for image in os.listdir(f"{self.root_dir}/{_class}"):
                self.img_paths.append(f"{self.root_dir}/{_class}/{image}")
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        This function is the most important part of our class. This returns the image
        and its corresponding label for the given index (idx).
        """
        img_name = self.img_paths[idx]

        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=T.Compose([
                                    T.ToTensor(),
                                    T.Resize((64, 64)),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))



plt.imshow(train_dataset[0][0].permute(1, 2, 0) * 0.5 + 0.5)
plt.show()


train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# The Generator of the GAN model
class Generator(nn.Module):
    def __init__(self, input_dim, n_classes, output_chs=3):
        super().__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.output_chs = output_chs

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim + self.n_classes, 512, 4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, self.output_chs, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, label):
        label = label.view(-1, self.n_classes, 1, 1)
        
        x = torch.concat([x , label], dim=1)
        
        x = self.deconv(x)
        
        return x[:, :, :64, :64]

# The Discriminator of the GAN model
class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        self.label_embedding = nn.Linear(self.n_classes, 64*64)
        
        self.features = nn.Sequential(
            # 64x64
            nn.Conv2d(4, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 32x32
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 16x16
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 8x8
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)       
        )

        self.layers = nn.Sequential(
            nn.Linear(8192, 1024), 
            nn.ReLU(),
            #nn.Dropout2d(0.5),

            nn.Linear(1024, 512), 
            nn.ReLU(),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        label = self.label_embedding(label).view(-1, 1, 64, 64)
        
        x = self.features(torch.concat((x, label), dim=1))

        x = torch.flatten(x, 1)

        return self.layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator(input_dim=1024, n_classes=10).to(device)
discriminator = Discriminator(n_classes=10).to(device)

print(generator(torch.zeros((32, 1024, 1, 1)).to(device), label=torch.zeros(32, 10).to(device)).shape)

print(discriminator(
    generator(torch.zeros((32, 1024, 1, 1)).to(device), label=torch.zeros(32, 10).to(device)),
    label=torch.zeros(32, 10).to(device)))

plt.imshow(generator(torch.zeros((32, 1024, 1, 1)).to(device), label=torch.zeros(32, 10).to(device))[0].detach().cpu().permute(1, 2, 0)  * 0.5 + 0.5)
plt.show()


# This time, we use BCELoss
loss_fn = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(4, 1024, 1, 1, device=device)
fixed_classes = torch.Tensor([
    [0],
    [1],
    [2],
    [3]
]).int()

real_label = 1.
fake_label = 0.

optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0003)

history_D = []
history_G = []

len_train_dataset = len(train_dataset)

len_train_dataloader = len(train_dataloader)

N_CLASSES = 10

num_epochs = 100

for epoch in range(num_epochs):
    total_loss_D = 0
    total_loss_G = 0

    discriminator.train()
    generator.train()

    for inputs, real_labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        real_labels = real_labels.int() #.to(device)
        real_labels_onehot = F.one_hot(torch.arange(N_CLASSES), N_CLASSES)[real_labels].to(device).float()

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        optimizerD.zero_grad()

        b_size = inputs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        output = discriminator(inputs, real_labels_onehot).view(-1)

        # Calculate loss on all-real batch
        errD_real = loss_fn(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 1024, 1, 1, device=device)

        # Generate fake image batch with G
        fake = generator(noise, real_labels_onehot)
        label.fill_(fake_label)

        # Classify all fake batch with D
        output = discriminator(fake.detach(), real_labels_onehot).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = loss_fn(output, label)

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake, real_labels_onehot).view(-1)

        # Calculate G's loss based on this output
        errG = loss_fn(output, label)

        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        total_loss_G += errG.item()
        total_loss_D += errD.item()

    generator.eval()

    fixed_classes_onehot = F.one_hot(torch.arange(N_CLASSES), N_CLASSES)[fixed_classes].to(device).float()
    generated_images = generator(fixed_noise, fixed_classes_onehot).detach().cpu()#.permute(1, 2, 0)

    plt.figure(1)
    for idx, img in enumerate(generated_images):
      plt.subplot(1, 4, idx+1)
      plt.imshow(img.permute(1, 2, 0)  * 0.5 + 0.5)
    plt.show()

    print(f"Epoch [{epoch+1}/{num_epochs}] Avg. Training Loss (D): {total_loss_D / len_train_dataloader}, Avg. Training Loss (G): {total_loss_G / len_train_dataloader}")

    history_D.append(total_loss_D / len_train_dataloader)
    history_G.append(total_loss_G / len_train_dataloader)