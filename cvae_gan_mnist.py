# cvae_gan_mnist.py
import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------
# Config / Hyperparameters
# -------------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 28
    channels = 1
    n_classes = 10
    latent_dim = 64
    hidden_dim = 512
    batch_size = 128
    lr = 2e-4
    betas = (0.5, 0.999)
    num_epochs = 30
    kl_anneal_steps = 10000  # optional: anneal KL weight
    recon_weight = 1.0      # pixel reconstruction weight
    feat_weight = 1.0       # discriminator feature matching weight
    adv_weight = 0.1        # adversarial loss weight (generator)
    save_dir = "./cvae_gan_checkpoints"

C = Config()

# -------------------------
# Utilities
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# One-hot helper
def one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes).float()

# -------------------------
# Models: Encoder, Decoder, Discriminator
# -------------------------
class Encoder(nn.Module):
    def __init__(self, img_dim:int, n_classes:int, hidden_dim:int, latent_dim:int):
        super().__init__()
        input_dim = img_dim + n_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_flat, y_onehot):
        # x_flat: (B, img_dim) flattened image
        h = torch.cat([x_flat, y_onehot], dim=1)
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, img_dim:int, n_classes:int, latent_dim:int, hidden_dim:int):
        super().__init__()
        input_dim = latent_dim + n_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, img_dim)

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.fc1(h))
        x_recon_logits = self.fc_out(h)
        x_recon = torch.sigmoid(x_recon_logits)  # pixel values in [0,1]
        return x_recon, x_recon_logits

class Discriminator(nn.Module):
    """
    Returns:
      prob: real/fake probability (B,1)
      feat: intermediate feature for perceptual loss
    """
    def __init__(self, img_dim:int, n_classes:int, hidden_dim:int):
        super().__init__()
        input_dim = img_dim + n_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_feat = nn.Linear(hidden_dim, hidden_dim//2)  # feature vector
        self.fc_out = nn.Linear(hidden_dim//2, 1)

    def forward(self, x_flat, y_onehot):
        h = torch.cat([x_flat, y_onehot], dim=1)
        h = F.leaky_relu(self.fc1(h), 0.2)
        feat = F.leaky_relu(self.fc_feat(h), 0.2)  # feature vector
        logit = self.fc_out(feat)
        prob = torch.sigmoid(logit)
        return prob, feat

# -------------------------
# Loss helpers
# -------------------------
def kl_divergence(mu, logvar):
    # computes KL(q||p) where p = N(0,I)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # per sample

# -------------------------
# Sampling / Saving helpers
# -------------------------
def save_images(x, filepath, nrow=8):
    utils.save_image(x, filepath, nrow=nrow, normalize=True)

# -------------------------
# Training loop
# -------------------------
def train():
    ensure_dir(C.save_dir)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    loader = DataLoader(train_ds, batch_size=C.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    img_dim = C.img_size * C.img_size * C.channels

    # Models
    E = Encoder(img_dim, C.n_classes, C.hidden_dim, C.latent_dim).to(C.device)
    G = Decoder(img_dim, C.n_classes, C.latent_dim, C.hidden_dim).to(C.device)
    D = Discriminator(img_dim, C.n_classes, C.hidden_dim).to(C.device)

    opt_eg = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=C.lr, betas=C.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=C.lr, betas=C.betas)

    bce_loss = nn.BCELoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')

    global_step = 0
    for epoch in range(1, C.num_epochs + 1):
        E.train(); G.train(); D.train()
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(C.device)            # (B,1,28,28)
            B = x.size(0)
            x_flat = x.view(B, -1)       # (B, img_dim)
            y = labels.to(C.device)
            y_onehot = one_hot(y, C.n_classes).to(C.device)

            # ---- 1) Train Discriminator ----
            # Real
            real_labels = torch.ones(B, 1, device=C.device)
            fake_labels = torch.zeros(B, 1, device=C.device)

            D_real_prob, D_real_feat = D(x_flat, y_onehot)
            loss_d_real = bce_loss(D_real_prob, real_labels)

            # Fake 1: reconstruction from encoder (use q(z|x))
            mu, logvar = E(x_flat, y_onehot)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
            x_recon, _ = G(z, y_onehot)
            x_recon_flat = x_recon.view(B, -1).detach()  # detach for D training
            D_recon_prob, _ = D(x_recon_flat, y_onehot)
            loss_d_recon = bce_loss(D_recon_prob, fake_labels)

            # Fake 2: sample from prior p(z) (optional, helps)
            z_prior = torch.randn(B, C.latent_dim, device=C.device)
            x_prior, _ = G(z_prior, y_onehot)
            x_prior_flat = x_prior.view(B, -1).detach()
            D_prior_prob, _ = D(x_prior_flat, y_onehot)
            loss_d_prior = bce_loss(D_prior_prob, fake_labels)

            loss_d = loss_d_real + 0.5 * (loss_d_recon + loss_d_prior)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ---- 2) Train Encoder+Decoder (VAE part + adversarial) ----
            # Compute KL
            mu, logvar = E(x_flat, y_onehot)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std

            x_recon, x_recon_logits = G(z, y_onehot)
            x_recon_flat = x_recon.view(B, -1)

            # Pixel reconstruction loss (BCE)
            recon_pixel = F.binary_cross_entropy(x_recon_flat, x_flat, reduction='sum') / B  # per-batch mean

            # KL
            kl = kl_divergence(mu, logvar).mean()  # mean over batch

            # Feature loss: match discriminator's intermediate features (use detached real feat)
            # Get features for real (detach no grad from D_real_feat)
            with torch.no_grad():
                _, D_real_feat_detached = D(x_flat, y_onehot)
            _, D_recon_feat = D(x_recon_flat, y_onehot)
            feat_loss = mse_loss(D_recon_feat, D_real_feat_detached)

            # Adversarial loss: try to fool D (label=1)
            D_recon_prob, _ = D(x_recon_flat, y_onehot)
            adv_loss = bce_loss(D_recon_prob, real_labels)

            # KL annealing (optional): linearly ramp up
            if C.kl_anneal_steps > 0:
                kl_weight = min(1.0, global_step / C.kl_anneal_steps)
            else:
                kl_weight = 1.0

            loss_eg = (C.recon_weight * recon_pixel) + (kl_weight * kl) \
                      + (C.feat_weight * feat_loss) + (C.adv_weight * adv_loss)

            opt_eg.zero_grad()
            loss_eg.backward()
            opt_eg.step()

            global_step += 1

            if batch_idx % 200 == 0:
                print(f"Epoch[{epoch}/{C.num_epochs}] Batch[{batch_idx}/{len(loader)}] "
                      f"loss_d={loss_d.item():.4f} loss_eg={loss_eg.item():.4f} recon={recon_pixel.item():.4f} kl={kl.item():.4f} adv={adv_loss.item():.4f} feat={feat_loss.item():.4f}")

        # Save checkpoint and sample images at each epoch
        E.eval(); G.eval()
        with torch.no_grad():
            # reconstruct few examples
            xs, ys = next(iter(loader))
            xs = xs.to(C.device)[:64]
            ys = ys.to(C.device)[:64]
            xs_flat = xs.view(xs.size(0), -1)
            yoh = one_hot(ys, C.n_classes).to(C.device)
            mu, logvar = E(xs_flat, yoh)
            z = mu  # use mean for neat reconstructions
            recon, _ = G(z, yoh)
            grid = torch.cat([xs, recon.view_as(xs)], dim=0)
            save_images(grid, os.path.join(C.save_dir, f"recon_epoch_{epoch}.png"), nrow=8)

            # class-conditional samples
            sample_list = []
            n_per_class = 8
            for c in range(C.n_classes):
                y_c = torch.tensor([c]*n_per_class, device=C.device)
                yoh_c = one_hot(y_c, C.n_classes).to(C.device)
                z_c = torch.randn(n_per_class, C.latent_dim, device=C.device)
                samp_c, _ = G(z_c, yoh_c)
                sample_list.append(samp_c)
            samples = torch.cat(sample_list, dim=0)
            save_images(samples.view(-1, C.channels, C.img_size, C.img_size),
                        os.path.join(C.save_dir, f"samples_epoch_{epoch}.png"),
                        nrow=n_per_class)

        # save model
        torch.save({
            "E": E.state_dict(),
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_eg": opt_eg.state_dict(),
            "opt_d": opt_d.state_dict()
        }, os.path.join(C.save_dir, f"cvae_gan_epoch_{epoch}.pth"))

    print("Training finished.")

if __name__ == "__main__":
    train()
