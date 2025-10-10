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
from tqdm.notebook import tqdm, trange # for Jupyter notebooks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, height=16, width=16):
        super(ConvNet, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1) # takes input of size (batch, in_channels, height, width) and ouputs (batch, 16, height, width)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # outputs (batch, 32, height, width)
        self.skip = nn.Identity()  # Placeholder for skip connection
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1) # expects concatenation of conv1 and conv2 outputs
        self.pool = nn.MaxPool2d(2, 2)

        # Task-specific heads
        self.input_size = int(64 * height/2 * width/2)  # Assuming input image size is 64x64

        self.grating_head = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 12 classes for gratings
        )

        self.natural_scene_head = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 118 classes for natural scenes
        )

    def forward(self, x):

        out_conv1 = self.relu(self.conv1(x))
        out_conv2 = self.relu(self.conv2(out_conv1))
        out_conv1_skip = self.skip(out_conv1)  # Skip connection
        
        out_concat = torch.cat((out_conv1_skip, out_conv2), dim=1)  # Concatenate along channel dimension
        out_conv3 = self.relu(self.conv3(out_concat))
        out_pool = self.pool(out_conv3)
        out_flat = out_pool.view(x.size(0), self.input_size)  # Flatten into shape (batch_size, input_size)

        # Task-specific outputs
        out_grating = self.grating_head(out_flat)
        out_natural_scene = self.natural_scene_head(out_flat)

        return out_grating, out_natural_scene
    
class GaborDataset(Dataset):
    def __init__(self, num_images=10000, img_size=32, orientations=10, frequencies=(0.08)):
        self.num_images = num_images
        self.img_size = img_size
        self.orientations = np.linspace(0, np.pi, orientations, endpoint=False) # generate orientations evenly spaced between 0 and 2pi
        self.frequencies = frequencies

        # Pre-generate all images and labels
        self.images = []
        self.labels = []
        for _ in range(num_images):
            orientation_idx = np.random.randint(0, len(self.orientations))
            orientation = self.orientations[orientation_idx]
            frequency = self.frequencies[0] if len(self.frequencies) == 1 else np.random.choice(self.frequencies)
            kernel = gabor_kernel(frequency, theta=orientation, sigma_x=15, sigma_y=15)
            gabor_img = np.real(kernel)
            gabor_img = (gabor_img - gabor_img.min()) / (gabor_img.max() - gabor_img.min()) * 255
            gabor_img = gabor_img.astype(np.uint8)
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            gabor_img_transformed = transform(Image.fromarray(gabor_img))
            self.images.append(gabor_img_transformed)
            self.labels.append(orientation_idx)
        self.images = torch.stack(self.images)
        self.labels = np.array(self.labels) + 10 # shift labels to avoid overlap with natural scenes labels

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
# --- Noise injection function ---
def add_noise_to_first_layer(model, noise_level=0.5):
    """Injects Gaussian noise ONLY into the first layer."""
    with torch.no_grad():
        noise = torch.randn_like(model.conv1.weight.data) * noise_level # returns tensor filled w/ random numbers from standard normal distribution * noise_level
        model.conv1.weight.data += noise
    return model

def add_noise_to_middle_layer(model, noise_level=0.5):
    """Injects Gaussian noise ONLY into the middle layer."""
    with torch.no_grad():
        noise = torch.randn_like(model.conv2.weight.data) * noise_level # returns tensor filled w/ random numbers from standard normal distribution * noise_level
        model.conv2.weight.data += noise
    return model

def add_noise_to_last_layer(model, noise_level=0.5):
    """Injects Gaussian noise ONLY into the final layer."""
    with torch.no_grad():
        noise = torch.randn_like(model.conv3.weight.data) * noise_level # returns tensor filled w/ random numbers from standard normal distribution * noise_level
        model.conv3.weight.data += noise
    return model

    
def train_model(model, dataloader, num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in trange(num_epochs, desc='Epochs'):
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader), desc=f'Batches in epoch #{epoch+1}', leave=False, total=len(dataloader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            out_grating, out_natural_scene = model(inputs)

            is_cifar = labels < 10
            is_gabor = labels >= 10

            loss = 0
            if is_cifar.any():
                loss += criterion(out_natural_scene[is_cifar], labels[is_cifar])
            if is_gabor.any():
                loss += criterion(out_grating[is_gabor], labels[is_gabor] - 10)  # adjust labels for gabor
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

        losses.append(running_loss / len(dataloader))

    return model, losses 

def evaluate_gabor_accuracy(model, gabor_test_loader):
    """Evaluates the Gabor head."""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in gabor_test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            gabor_outputs, _ = model(images) # Only use Gabor output
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
            _, cifar_outputs = model(images) # Only use cifar output
            _, predicted = torch.max(cifar_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def plot_losses(losses):
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    
    axs.plot(losses)

    axs.set_xlabel('Epoch')
    axs.set_ylabel('Cross-Entropy Loss')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    axs.set_title('Training Loss')
    plt.show()

    return fig, axs