# cvae_gan_conv64.py
import os
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------
# Config
# -------------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64
    channels = 3
    n_classes = 10          # e.g. CIFAR10; change for your dataset
    latent_dim = 128
    batch_size = 64
    lr = 2e-4
    betas = (0.5, 0.999)
    num_epochs = 20
    recon_weight = 1.0
    feat_weight = 1.0
    adv_weight = 0.1
    save_dir = "./cvae_gan_conv_checkpoints"

C = Config()

# -------------------------
# Utility helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def one_hot(labels, num_classes, device=None):
    return F.one_hot(labels, num_classes).float().to(device if device is not None else labels.device)

# -------------------------
# VAE Encoder / Decoder / VAE (conv)
# -------------------------
class Encoder(nn.Module):
    """
    Conv encoder that maps (B,3,64,64) -> mu, log_var (both (B, latent_dim))
    We accept an optional label embedding which is concatenated to flattened conv-features.
    """
    def __init__(self, output_dim, n_classes=None, emb_dim=32):
        super().__init__()
        self.output_dim = output_dim
        self.n_classes = n_classes
        self.emb_dim = emb_dim if n_classes is not None else 0

        self.features = nn.Sequential(
            # input: 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),                           # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),                           # 16x16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),                           # 8x8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),                           # 4x4
        )

        # after flatten we have 256 * 4 * 4 = 4096
        flattened_dim = 256 * 4 * 4
        self.layers = nn.Sequential(
            nn.Linear(flattened_dim + self.emb_dim, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        # map from 512 -> latent
        self.layers_mu = nn.Linear(512, self.output_dim)
        self.layers_log_variance = nn.Linear(512, self.output_dim)

        if n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, self.emb_dim)

    def forward(self, x, y=None):
        # x: (B,3,64,64)
        h = self.features(x)
        h = torch.flatten(h, start_dim=1)  # (B, 4096)
        if self.n_classes is not None:
            assert y is not None, "Labels required for conditional encoder"
            y_emb = self.label_emb(y)  # (B, emb_dim)
            h = torch.cat([h, y_emb], dim=1)
        h = self.layers(h)            # (B,512)
        mu = self.layers_mu(h)        # (B,latent_dim)
        log_var = self.layers_log_variance(h)
        return mu, log_var

class Decoder(nn.Module):
    """
    Decoder: z (+ label embedding) -> reconstructed image (B,3,64,64)
    Expect input z of dimension latent_dim (and y labels if conditional).
    """
    def __init__(self, input_dim, n_classes=None, emb_dim=32, output_chs=3):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.emb_dim = emb_dim if n_classes is not None else 0
        self.output_chs = output_chs

        # FC to expand to 4096
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.emb_dim, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
        )

        # Deconv stack: 4x4 -> 8 -> 16 -> 32 -> 64
        # Use kernel_size=4, stride=2, padding=1 to double spatial dims.
        self.deconv = nn.Sequential(
            # input expected shape: (B,256,4,4)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, self.output_chs, kernel_size=3, padding=1),
            nn.Sigmoid(),  # outputs in [0,1]
        )

        if n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, self.emb_dim)

    def forward(self, z, y=None):
        # z: (B, latent_dim)
        if self.n_classes is not None:
            assert y is not None, "Labels required for conditional decoder"
            y_emb = self.label_emb(y)
            h = torch.cat([z, y_emb], dim=1)
        else:
            h = z
        h = self.fc(h)  # (B, 4096)
        h = h.view(-1, 256, 4, 4)  # (B,256,4,4)
        x = self.deconv(h)         # (B,3,64,64)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim, n_classes=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(output_dim=latent_dim, n_classes=n_classes)
        self.decoder = Decoder(input_dim=latent_dim, n_classes=n_classes)

    def encode(self, x, y):
        return self.encoder(x, y)

    def decode(self, z, y):
        return self.decoder(z, y)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, y)
        return recon, mu, log_var

# -------------------------
# Discriminator (conv conditional)
# -------------------------
class Discriminator(nn.Module):
    """
    Conv discriminator returning probability and an intermediate feature vector.
    Conditioning implemented by concatenating a label-channel map.
    """
    def __init__(self, n_classes=None, emb_dim=32):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim if n_classes is not None else 0

        in_ch = 3 + (1 if n_classes is not None else 0)  # we'll append one channel of label map
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),    # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),   # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),   # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # feature vector
        self.fc_feat = nn.Linear(512 * 4 * 4, 1024)
        self.fc_out = nn.Linear(1024, 1)  # real/fake logit

        if n_classes is not None:
            self.label_emb = nn.Embedding(n_classes, emb_dim)

    def forward(self, x, y=None):
        # x: (B,3,64,64)
        if self.n_classes is not None:
            assert y is not None
            # simple: create a 1-channel spatial map from label embedding and concat
            # alternative: replicate embedding spatially
            b = x.size(0)
            # embed labels to a scalar map (or small channel)
            # We'll embed -> scalar per label then expand to HxW
            emb = self.label_emb(y)                      # (B, emb_dim)
            # reduce emb to single-channel map via linear
            map_scalar = emb.mean(dim=1, keepdim=True)   # (B,1)
            map_map = map_scalar.view(b,1,1,1).expand(-1,1,x.size(2),x.size(3))  # (B,1,H,W)
            x_in = torch.cat([x, map_map], dim=1)
        else:
            x_in = x

        h = self.conv(x_in)  # (B,512,4,4)
        h_flat = h.view(h.size(0), -1)
        feat = F.leaky_relu(self.fc_feat(h_flat), 0.2)  # feature vector for perceptual loss
        logit = self.fc_out(feat)
        prob = torch.sigmoid(logit)
        return prob, feat

# -------------------------
# Loss helpers
# -------------------------
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # per-sample

# -------------------------
# Training loop
# -------------------------
def train():
    ensure_dir(C.save_dir)

    # Data: CIFAR10 resized to 64x64 (example). Replace with your dataset.
    transform = transforms.Compose([
        transforms.Resize(C.img_size),
        transforms.ToTensor(),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    loader = DataLoader(train_ds, batch_size=C.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Models
    vae = VAE(latent_dim=C.latent_dim, n_classes=C.n_classes).to(C.device)
    D = Discriminator(n_classes=C.n_classes).to(C.device)

    opt_eg = torch.optim.Adam(list(vae.parameters()), lr=C.lr, betas=C.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=C.lr, betas=C.betas)

    bce = nn.BCELoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')

    global_step = 0
    for epoch in range(1, C.num_epochs + 1):
        vae.train(); D.train()
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(C.device)
            labels = labels.to(C.device)
            B = x.size(0)

            real_labels = torch.ones(B,1,device=C.device)
            fake_labels = torch.zeros(B,1,device=C.device)

            # ----------------------
            # Train Discriminator
            # ----------------------
            # real
            D_real_prob, D_real_feat = D(x, labels)
            loss_d_real = bce(D_real_prob, real_labels)

            # fake: reconstruction from VAE encoder (q(z|x))
            mu, log_var = vae.encode(x, labels)
            z = vae.reparameterize(mu, log_var)
            x_recon = vae.decode(z, labels).detach()
            D_recon_prob, _ = D(x_recon, labels)
            loss_d_recon = bce(D_recon_prob, fake_labels)

            # fake: sample from prior
            z_prior = torch.randn(B, C.latent_dim, device=C.device)
            # sample also requires labels to be conditional (we use same labels)
            x_prior = vae.decode(z_prior, labels).detach()
            D_prior_prob, _ = D(x_prior, labels)
            loss_d_prior = bce(D_prior_prob, fake_labels)

            loss_d = loss_d_real + 0.5 * (loss_d_recon + loss_d_prior)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ----------------------
            # Train VAE (encoder + decoder) with VAE + adv + feature matching
            # ----------------------
            mu, log_var = vae.encode(x, labels)
            z = vae.reparameterize(mu, log_var)
            x_recon = vae.decode(z, labels)
            # pixel-wise reconstruction (MSE for color images)
            recon_pixel = F.mse_loss(x_recon, x, reduction='sum') / B
            kl = kl_divergence(mu, log_var).mean()

            # feature matching: use D features
            with torch.no_grad():
                _, D_real_feat_det = D(x, labels)
            _, D_recon_feat = D(x_recon, labels)
            feat_loss = mse(D_recon_feat, D_real_feat_det)

            # adversarial loss
            D_recon_prob, _ = D(x_recon, labels)
            adv_loss = bce(D_recon_prob, real_labels)

            loss_eg = (C.recon_weight * recon_pixel) + kl + (C.feat_weight * feat_loss) + (C.adv_weight * adv_loss)

            opt_eg.zero_grad()
            loss_eg.backward()
            opt_eg.step()

            global_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch[{epoch}/{C.num_epochs}] Batch[{batch_idx}/{len(loader)}] "
                      f"loss_d={loss_d.item():.4f} loss_eg={loss_eg.item():.4f} recon={recon_pixel.item():.4f} kl={kl.item():.4f} adv={adv_loss.item():.4f} feat={feat_loss.item():.4f}")

        # Save samples & checkpoint each epoch
        vae.eval()
        with torch.no_grad():
            xs, ys = next(iter(loader))
            xs = xs.to(C.device)[:32]; ys = ys.to(C.device)[:32]
            recon, _, _ = vae(xs, ys)
            grid = torch.cat([xs, recon], dim=0)
            utils.save_image(grid, os.path.join(C.save_dir, f"recon_epoch_{epoch}.png"), nrow=8, normalize=True)

            # conditional samples: for each class generate n_per_class samples
            n_per = 8
            sample_list = []
            for c in range(C.n_classes):
                y_c = torch.tensor([c]*n_per, device=C.device)
                z_c = torch.randn(n_per, C.latent_dim, device=C.device)
                samp_c = vae.decode(z_c, y_c)
                sample_list.append(samp_c)
            samples = torch.cat(sample_list, dim=0)
            utils.save_image(samples, os.path.join(C.save_dir, f"samples_epoch_{epoch}.png"), nrow=n_per, normalize=True)

        torch.save({
            "vae": vae.state_dict(),
            "D": D.state_dict(),
            "opt_eg": opt_eg.state_dict(),
            "opt_d": opt_d.state_dict()
        }, os.path.join(C.save_dir, f"cvae_gan_epoch_{epoch}.pth"))

    print("Training finished.")

if __name__ == "__main__":
    train()
