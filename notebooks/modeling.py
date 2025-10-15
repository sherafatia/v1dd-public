import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from skimage.filters import gabor_kernel
from PIL import Image
import matplotlib.pyplot as plt

# from tqdm import tqdm
from tqdm.notebook import tqdm, trange  # for Jupyter notebooks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.skip = nn.Identity()  # Placeholder for skip connection
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
            0, 180, orientations, endpoint=False
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


# --- Noise injection function ---
def add_noise_to_first_layer(model, noise_level=0.5):
    """Injects Gaussian noise ONLY into the first layer."""
    with torch.no_grad():
        noise = (
            torch.randn_like(model.conv1.weight.data) * noise_level
        )  # returns tensor filled w/ random numbers from standard normal distribution * noise_level
        model.conv1.weight.data += noise
    return model


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
    combined_val_loader=None,
    gabor_val_loader=None,
    cifar_val_loader=None,
    num_epochs=10,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    gabor_val_accuracies = []
    cifar_val_accuracies = []
    for epoch in trange(num_epochs, desc="Epochs"):

        for _, data in tqdm(
            enumerate(train_loader),
            desc=f"Batches in epoch #{epoch+1}",
            leave=False,
            total=len(train_loader),
        ):
            running_loss = 0.0
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

            running_loss += (
                train_loss.item()
                if isinstance(train_loss, torch.Tensor)
                else train_loss
            )
            train_losses.append(running_loss)

            running_loss = 0.0
            if combined_val_loader:
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
                            val_loss += criterion(
                                out_natural_scene[is_cifar], labels[is_cifar]
                            )
                        if is_gabor.any():
                            val_loss += criterion(
                                out_grating[is_gabor], labels[is_gabor] - 10
                            )  # adjust labels for gabor

                        running_loss += (
                            val_loss.item()
                            if isinstance(val_loss, torch.Tensor)
                            else val_loss
                        )

                val_losses.append(running_loss / len(combined_val_loader))

            if gabor_val_loader:
                gabor_accuracy = evaluate_gabor_accuracy(model, gabor_val_loader)
                gabor_val_accuracies.append(gabor_accuracy)

            if cifar_val_loader:
                cifar_accuracy = evaluate_cifar_accuracy(model, cifar_val_loader)
                cifar_val_accuracies.append(cifar_accuracy)

            train_loss.backward()
            optimizer.step()

        # train_losses.append(running_loss / len(train_loader))

    return model, train_losses, val_losses, gabor_val_accuracies, cifar_val_accuracies


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
