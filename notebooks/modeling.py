import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from skimage.filters import gabor_kernel
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import random

# from tqdm import tqdm
from tqdm.notebook import tqdm, trange  # for Jupyter notebooks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoisyIdentity(nn.Module):
    def __init__(self, noise_level=0.1):
        super(NoisyIdentity, self).__init__()
        self.noise_level = noise_level

    def forward(self, x):
        if not self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        else:
            return x


class ConvNet(nn.Module):
    def __init__(self, in_channels=1, height=32, width=32):
        super(ConvNet, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, padding=1
        )  # takes input of size (batch, in_channels, height, width) and outputs (batch, 16, height, width)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16, 16, kernel_size=3, padding=1
        )  # outputs (batch, 16, height, width)
        self.skip = NoisyIdentity()  # Placeholder for skip connection
        self.pool = nn.MaxPool2d(2, 2)

        # Task-specific heads
        self.input_size = int(
            32 * height / 2 * width / 2
        )  # Assuming input image size is 64x64
        self.grating_head = nn.Linear(self.input_size, 10)  # 10 classes for gratings
        self.natural_scene_head = nn.Linear(
            self.input_size, 10
        )  # 10 classes for natural scenes

    def forward(self, x):

        out_conv1 = self.relu(self.conv1(x))
        out_conv2 = self.relu(self.conv2(out_conv1))
        out_conv1_skip = self.skip(out_conv1)  # Skip connection

        out_concat = torch.cat(
            (out_conv1_skip, out_conv2), dim=1
        )  # Concatenate along channel dimension
        # out_conv3 = self.relu(self.conv3(out_concat))
        out_pool = self.pool(out_concat)
        out_flat = out_pool.view(
            x.size(0), self.input_size
        )  # Flatten into shape (batch_size, input_size)

        # Task-specific outputs
        out_grating = self.grating_head(out_flat)
        out_natural_scene = self.natural_scene_head(out_flat)

        return out_grating, out_natural_scene


class ConvNet_NoSkip(nn.Module):
    def __init__(self, in_channels=1, height=32, width=32):
        super(ConvNet_NoSkip, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, padding=1
        )  # takes input of size (batch, in_channels, height, width) and outputs (batch, 16, height, width)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16, 16, kernel_size=3, padding=1
        )  # outputs (batch, 16, height, width)
        self.pool = nn.MaxPool2d(2, 2)

        # Task-specific heads
        self.input_size = int(
            16 * height / 2 * width / 2
        )  # Assuming input image size is 64x64
        self.grating_head = nn.Linear(self.input_size, 10)  # 10 classes for gratings
        self.natural_scene_head = nn.Linear(
            self.input_size, 10
        )  # 10 classes for natural scenes

    def forward(self, x):

        out_conv1 = self.relu(self.conv1(x))
        out_conv2 = self.relu(self.conv2(out_conv1))

        out_pool = self.pool(out_conv2)
        out_flat = out_pool.view(
            x.size(0), self.input_size
        )  # Flatten into shape (batch_size, input_size)

        # Task-specific outputs
        out_grating = self.grating_head(out_flat)
        out_natural_scene = self.natural_scene_head(out_flat)

        return out_grating, out_natural_scene


class GaborDataset(Dataset):
    def __init__(
        self,
        num_images=10000,
        img_size=32,
        orientations=10,
        frequencies=[0.08, 0.04],
        add_noise=False,
        noise_level=0.1,
    ):
        self.num_images = num_images
        self.img_size = img_size
        self.orientations = np.linspace(
            0, np.pi, orientations, endpoint=False
        )  # generate orientations evenly spaced between 0 and 180 degrees
        self.frequencies = frequencies

        # Pre-generate all images and labels
        assert (
            self.num_images % (len(self.orientations) * len(self.frequencies)) == 0
        ), "num_images must be divisible by the product of number of orientations and frequencies"

        img_per_orientation = self.num_images // len(np.unique(self.orientations))
        img_per_frequency = img_per_orientation // len(self.frequencies)

        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        images = []
        labels = []
        for theta in self.orientations:
            for freq in self.frequencies:
                for _ in range(img_per_frequency):
                    kernel = np.real(
                        gabor_kernel(
                            frequency=freq, theta=theta, sigma_x=15, sigma_y=15
                        )
                    )
                    kernel = (
                        (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
                    )  # Normalize to [0, 255]
                    kernel = kernel.astype(np.uint8)
                    img = transform(Image.fromarray(kernel))
                    images.append(img)
                    labels.append(
                        np.where(self.orientations == theta)[0][0]
                    )  # Label by orientation

        if add_noise:
            noise = torch.randn_like(torch.stack(images)) * noise_level
            images = [img + n for img, n in zip(images, noise)]

        self.images = torch.stack(images)  # Shape: (num_images, 1, img_size, img_size)
        self.labels = np.array(labels) + 10  # Shape: (num_images,)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class GaborDatasetNoisy(Dataset):
    def __init__(
        self,
        num_images=1000,
        img_size=32,
        orientations=10,
        frequencies=[0.08, 0.04],
        noise_level=0.3,
        orientation_jitter=0.1,
        frequency_jitter=0.01,
        sigma_range=(10, 20),
        offset_jitter=5,
        seed=5,
    ):
        super().__init__()
        self.num_images = num_images
        self.img_size = img_size
        self.num_orientations = orientations
        self.orientations = np.linspace(
            0, -np.pi, orientations, endpoint=False
        )  # -np.pi because skimage's gabor_kernel uses a different convention
        self.frequencies = frequencies
        self.orientation_jitter = orientation_jitter
        self.frequency_jitter = frequency_jitter
        self.sigma_range = sigma_range
        self.noise_level = noise_level
        self.offset_jitter = offset_jitter

        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        images = []
        labels = []
        for idx in range(self.num_images):
            class_id = idx % self.num_orientations
            base_orientation = self.orientations[class_id]
            base_frequency = random.choice(self.frequencies)

            jittered_orientation = base_orientation + self.rng.normal(
                0, self.orientation_jitter
            )
            jittered_frequency = base_frequency + self.rng.normal(
                0, self.frequency_jitter
            )
            sigma_x = self.rng.uniform(self.sigma_range[0], self.sigma_range[1])
            sigma_y = self.rng.uniform(self.sigma_range[0], self.sigma_range[1])

            kernel = np.real(
                gabor_kernel(
                    frequency=jittered_frequency,
                    theta=jittered_orientation,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                )
            )
            offset = random.randint(-self.offset_jitter, self.offset_jitter)
            kernel = np.roll(kernel, shift=(offset, offset), axis=(0, 1))

            kernel = (
                (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
            )  # Normalize to [0, 255]
            kernel = kernel.astype(np.uint8)
            img = self.transform(Image.fromarray(kernel))
            noise = torch.randn_like(img) * self.noise_level
            img_noisy = img + noise

            label = class_id + 10  # Adjust label to be in the range [10, 19]

            images.append(img_noisy)
            labels.append(label)

        self.images = torch.stack(images)  # Shape: (num_images, 1, img_size, img_size)
        self.labels = np.array(labels)  # Shape: (num_images,)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# --- Noise injection function ---
# def add_noise_to_first_layer(model, noise_level=0.5):
#     """Injects Gaussian noise ONLY into the first layer."""
#     with torch.no_grad():
#         noise = (
#             torch.randn_like(model.conv1.weight.data) * noise_level
#         )  # returns tensor filled w/ random numbers from standard normal distribution * noise_level
#         model.conv1.weight.data += noise
#     return model


def add_noise_to_middle_layer(model, noise_level=0.5):
    """Injects Gaussian noise ONLY into the middle layer."""
    with torch.no_grad():
        noise = (
            torch.randn_like(model.conv2.weight.data) * noise_level
        )  # returns tensor filled w/ random numbers from standard normal distribution * noise_level
        model.conv2.weight.data += noise
    return model


def train_model(
    model,
    train_loader,
    batch_size=64,
    probe_size=256,
    val_dataset=None,
    num_epochs=10,
    early_stopping=False,
    tag=None,
):

    if tag is None:
        tag = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

    if not os.path.exists(f"./model_states/{tag}"):
        os.makedirs(f"./model_states/{tag}")
    print(f"Training model with tag: {tag}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=1e-6
    )

    train_losses_epoch, val_losses_epoch = [], []
    train_losses, val_losses = [], []
    gabor_val_accuracies = []
    cifar_val_accuracies = []
    for epoch in trange(num_epochs, desc="Epochs"):

        running_val_loss = 0.0
        combined_val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True
        )
        with torch.no_grad():
            for data in combined_val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                out_grating, out_natural_scene = model(
                    images
                )  # Only use combined output

                is_cifar = labels < 10
                is_gabor = labels >= 10

                val_loss = 0
                if is_cifar.any():
                    cifar_preds = out_natural_scene[is_cifar]
                    cifar_labels = labels[is_cifar]
                    val_loss += criterion(cifar_preds, cifar_labels)

                if is_gabor.any():
                    gabor_preds = out_grating[is_gabor]
                    gabor_labels = labels[is_gabor] - 10
                    val_loss += criterion(gabor_preds, gabor_labels)

                running_val_loss += (
                    val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
                )
        val_losses_epoch.append(running_val_loss / len(combined_val_loader))
        if epoch > 2 and early_stopping is True:
            if val_losses_epoch[-1] > val_losses_epoch[-2] + 0.05:
                print(
                    "Early stopping triggered: validation loss increased significantly {:.4f} -> {:.4f}.".format(
                        val_losses_epoch[-2], val_losses_epoch[-1]
                    )
                )
                torch.save(
                    model.state_dict(),
                    f"./model_states/{tag}/model_epoch_{epoch+1}_best.pth",
                )
                df_losses = pd.DataFrame(
                    {
                        "train_loss": train_losses,
                        "val_loss": val_losses,
                        "gabor_val_accuracy": gabor_val_accuracies,
                        "cifar_val_accuracy": cifar_val_accuracies,
                    }
                )
                df_losses.to_csv(f"./model_states/losses_{tag}.csv", index=False)
                return (
                    model,
                    train_losses,
                    val_losses,
                    train_losses_epoch,
                    val_losses_epoch,
                    gabor_val_accuracies,
                    cifar_val_accuracies,
                )

        running_train_loss = 0.0
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                out_grating, out_natural_scene = model(
                    images
                )  # Only use combined output

                is_cifar = labels < 10
                is_gabor = labels >= 10

                train_loss = 0
                if is_cifar.any():
                    cifar_preds = out_natural_scene[is_cifar]
                    cifar_labels = labels[is_cifar]
                    train_loss += criterion(cifar_preds, cifar_labels)

                if is_gabor.any():
                    gabor_preds = out_grating[is_gabor]
                    gabor_labels = labels[is_gabor] - 10
                    train_loss += criterion(gabor_preds, gabor_labels)

                running_train_loss += (
                    train_loss.item()
                    if isinstance(train_loss, torch.Tensor)
                    else train_loss
                )
        train_losses_epoch.append(running_train_loss / len(train_loader))

        running_training_loss = 0.0
        for idx, data in tqdm(
            enumerate(train_loader),
            desc=f"Batches in epoch #{epoch+1}",
            leave=False,
            total=len(train_loader),
        ):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            out_grating, out_natural_scene = model(inputs)

            is_cifar = labels < 10
            is_gabor = labels >= 10

            train_loss = 0
            if is_cifar.any():
                train_loss += criterion(out_natural_scene[is_cifar], labels[is_cifar])
            if is_gabor.any():
                train_loss += criterion(
                    out_grating[is_gabor], labels[is_gabor] - 10
                )  # adjust labels for gabor

            train_losses.append(
                train_loss.item()
                if isinstance(train_loss, torch.Tensor)
                else train_loss
            )
            running_training_loss += train_losses[-1]

            probe_val_loss = 0.0
            if val_dataset is not None:
                indices = torch.randperm(len(val_dataset))[
                    : min(probe_size, len(val_dataset))
                ].tolist()
                probe = Subset(val_dataset, indices)

                combined_val_loader = DataLoader(
                    probe, batch_size=batch_size, shuffle=True
                )
                correct_gabor, correct_cifar = 0, 0
                total_gabor, total_cifar = 0, 0
                with torch.no_grad():
                    for data in combined_val_loader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        out_grating, out_natural_scene = model(
                            images
                        )  # Only use combined output

                        is_cifar = labels < 10
                        is_gabor = labels >= 10

                        val_loss = 0
                        if is_cifar.any():
                            cifar_preds = out_natural_scene[is_cifar]
                            cifar_labels = labels[is_cifar]
                            val_loss += criterion(cifar_preds, cifar_labels)
                            _, predicted = torch.max(cifar_preds.data, 1)
                            total_cifar += cifar_labels.size(0)
                            correct_cifar += (predicted == cifar_labels).sum().item()

                        if is_gabor.any():
                            gabor_preds = out_grating[is_gabor]
                            gabor_labels = labels[is_gabor] - 10
                            val_loss += criterion(gabor_preds, gabor_labels)
                            _, predicted = torch.max(gabor_preds.data, 1)
                            total_gabor += gabor_labels.size(0)
                            correct_gabor += (predicted == gabor_labels).sum().item()

                        probe_val_loss += (
                            val_loss.item()
                            if isinstance(val_loss, torch.Tensor)
                            else val_loss
                        )

                cifar_val_accuracies.append(
                    100 * correct_cifar / total_cifar if total_cifar > 0 else 0
                )
                gabor_val_accuracies.append(
                    100 * correct_gabor / total_gabor if total_gabor > 0 else 0
                )

                val_losses.append(probe_val_loss / len(combined_val_loader))

            train_loss.backward()
            optimizer.step()
            scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}]"
            f"Training Loss: {train_losses_epoch[-1]:.4f}"
            f"Validation Loss: {val_losses_epoch[-1]:.4f}"
        )

        # clear and store model after each epoch
        torch.cuda.empty_cache()

        torch.save(
            model.state_dict(), f"./model_states/{tag}/model_epoch_{epoch+1}.pth"
        )

    df_losses = pd.DataFrame(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "gabor_val_accuracy": gabor_val_accuracies,
            "cifar_val_accuracy": cifar_val_accuracies,
        }
    )
    df_losses.to_csv(f"./model_states/losses_{tag}.csv", index=False)

    return (
        model,
        train_losses,
        val_losses,
        train_losses_epoch,
        val_losses_epoch,
        gabor_val_accuracies,
        cifar_val_accuracies,
    )


def evaluate_gabor_accuracy(model, gabor_test_loader):
    """Evaluates the Gabor head."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in gabor_test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            gabor_outputs, _ = model(images)  # Only use Gabor output
            _, predicted = torch.max(gabor_outputs.data, 1)
            labels = labels - 10
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def evaluate_cifar_accuracy(model, cifar_test_loader):
    """Evaluates the Cifar head."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in cifar_test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _, cifar_outputs = model(images)  # Only use cifar output
            _, predicted = torch.max(cifar_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def plot_losses(losses):
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    axs.plot(losses)

    axs.set_xlabel("Epoch")
    axs.set_ylabel("Cross-Entropy Loss")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    axs.set_title("Training Loss")
    plt.show()

    return fig, axs
