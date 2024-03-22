import os

import torch
import torch.nn as nn


from torch import autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    """A simplified self-attention layer for illustrative purposes."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.shape
        q = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W * H)
        v = self.value(x).view(B, -1, W * H).permute(0, 2, 1)

        attention = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        return out + x  # Add skip connection

class Generator(nn.Module):
    def __init__(self, z_dim, text_embedding_dim, img_size, channels):
        super(Generator, self).__init__()
        self.init_size = img_size // 8
        # Initial linear layer
        self.l1 = nn.Sequential(nn.Linear(z_dim + text_embedding_dim, 320 * self.init_size ** 2))

        # Convolutional blocks, trying to mimic the structure from the provided config
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(320),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(320, 320, 3, stride=1, padding=1),
                nn.BatchNorm2d(320),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(320) if img_size // 2**3 in [4, 2] else nn.Identity(),
            ),
            nn.Sequential(
                nn.Conv2d(320, 640, 3, stride=1, padding=1),
                nn.BatchNorm2d(640),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(640) if img_size // 2**2 in [4, 2] else nn.Identity(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(640, 1280, 3, stride=1, padding=1),
                nn.BatchNorm2d(1280),
                nn.LeakyReLU(0.2, inplace=True),
                SelfAttention(1280) if img_size // 2**1 in [4, 2] else nn.Identity(),
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(1280, 1280, 3, stride=1, padding=1),
                nn.BatchNorm2d(1280),
                nn.LeakyReLU(0.2, inplace=True),
                # Consider adding SelfAttention here as well if it fits your architectural needs
            ),
        ])

        # Final convolution to output image
        self.final_conv = nn.Sequential(
            nn.Conv2d(1280, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, text_embedding):
        x = torch.cat([z, text_embedding], dim=1)
        out = self.l1(x)
        out = out.view(out.shape[0], 320, self.init_size, self.init_size)
        for block in self.conv_blocks:
            out = block(out)
        img = self.final_conv(out)
        return img
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(512, 1, 3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),  # Add this line
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)


# Sampler class for generating fake data samples
class Sampler(object):
    def __init__(self, generator):
        self.G = generator

    def sample(self, n):
        z = torch.randn(n, self.G.latent_dim, device=device)
        samples = self.G(z)
        return samples

# Custom Dataset class as defined previously
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, image_paths, latent_folder, text_embedding_folder):
        self.image_paths = image_paths
        self.latent_folder = latent_folder
        self.text_embedding_folder = text_embedding_folder
        # Ensure there's at least one image, or add error handling for an empty list
        assert len(self.image_paths) > 0, "The dataset cannot be empty."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Use modulo to cycle through the dataset
        idx = idx % len(self.image_paths)

        image_path = self.image_paths[idx]
        base_name = os.path.basename(image_path).replace('.png', '')
        latent_path = os.path.join(self.latent_folder, f"{base_name}.pt")
        text_embedding_path = os.path.join(self.text_embedding_folder, f"{base_name}.pt")

        real_data = torch.load(latent_path)
        text_embedding = torch.load(text_embedding_path)

        return real_data, text_embedding
# Custom function for calculating gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Adjust the shape of 'fake' to match the output of D
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def merge_state_dicts(src_state_dict, dest_state_dict, alpha=0.5):
    """
    Merges tensors from src_state_dict into dest_state_dict with a blending factor alpha.

    Parameters:
    - src_state_dict: The source state dictionary.
    - dest_state_dict: The destination state dictionary.
    - alpha: The blending factor, where 0 <= alpha <= 1. A value of 1 means the source tensor fully
             replaces the destination tensor, and a value of 0 means the destination tensor is unchanged.

    Returns:
    - A new state_dict that is the result of blending src_state_dict into dest_state_dict.
    """
    merged_state_dict = dest_state_dict.copy()
    used_dest_tensors = set()  # Tracks destination tensors that have already been merged into

    for src_name, src_tensor in src_state_dict.items():
        for dest_name, dest_tensor in dest_state_dict.items():
            # Check if the destination tensor has already been merged into
            if dest_name in used_dest_tensors:
                continue  # Skip this destination tensor if it has already been used for merging

            # Proceed with merging if dimensions match and the destination tensor hasn't been used yet
            if src_tensor.size() == dest_tensor.size():
                # Blend the source tensor into the destination tensor
                merged_tensor = alpha * src_tensor + (1 - alpha) * dest_tensor
                merged_state_dict[dest_name] = merged_tensor
                used_dest_tensors.add(dest_name)  # Mark this destination tensor as used
                print(f"Merged {src_name} into {dest_name} based on matching dimensions.")
                break  # Break to ensure this source tensor doesn't attempt to merge into another destination tensor

    return merged_state_dict