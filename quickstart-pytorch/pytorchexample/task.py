"""pytorchexample: A Flower / PyTorch app."""

from pathlib import Path

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, MetricRecord
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class Net(nn.Module):
    """Lightweight Vision Transformer for CIFAR-10 (32x32)."""

    def __init__(self):
        super().__init__()
        image_size = 32
        patch_size = 4
        embed_dim = 192
        num_heads = 4
        depth = 6
        mlp_dim = 384
        num_classes = 10
        epochs = 20

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


LOCAL_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
IMAGE_TRANSFORMS = Compose(
    [Resize((32, 32)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def _resolve_dataset_dir(dataset_path: str | None) -> Path:
    """Return dataset path from run config or default workspace location."""
    if dataset_path:
        return Path(dataset_path).expanduser().resolve()
    return LOCAL_DATASET_DIR


def _partition_indices(
    dataset_len: int, partition_id: int, num_partitions: int, seed: int = 42
) -> list[int]:
    """Create deterministic IID partitions by shuffling then slicing."""
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_len, generator=generator).tolist()

    base_size = dataset_len // num_partitions
    remainder = dataset_len % num_partitions

    start = partition_id * base_size + min(partition_id, remainder)
    length = base_size + (1 if partition_id < remainder else 0)
    end = start + length
    return shuffled[start:end]


def _build_local_dataset(dataset_path: str | None) -> ImageFolder:
    dataset_dir = _resolve_dataset_dir(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. "
            "Set 'dataset-path' in run-config or place data in ../dataset."
        )
    return ImageFolder(root=str(dataset_dir), transform=IMAGE_TRANSFORMS)


def load_data(
    partition_id: int, num_partitions: int, batch_size: int, dataset_path: str | None = None
):
    """Load one client partition from local image dataset."""
    dataset = _build_local_dataset(dataset_path)
    indices = _partition_indices(len(dataset), partition_id, num_partitions)
    client_subset = Subset(dataset, indices)

    train_size = int(0.8 * len(client_subset))
    val_size = len(client_subset) - train_size
    generator = torch.Generator().manual_seed(42 + partition_id)
    train_subset, val_subset = random_split(
        client_subset, [train_size, val_size], generator=generator
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset_path: str | None = None):
    """Load centralized holdout split for global evaluation."""
    dataset = _build_local_dataset(dataset_path)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(123)
    _, test_subset = random_split(dataset, [train_size, test_size], generator=generator)
    return DataLoader(test_subset, batch_size=128)


def _unpack_batch(batch):
    """Support dict batches (HF-style) and tuple batches (ImageFolder)."""
    if isinstance(batch, dict):
        return batch["img"], batch["label"]
    images, labels = batch
    return images, labels


def train(
    net,
    trainloader,
    epochs,
    lr,
    device,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = torch.tensor(0.0, device=device)
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += torch.sum((local_weights - global_weights) ** 2)
                loss = loss + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def global_evaluate(
    server_round: int, arrays: ArrayRecord, dataset_path: str | None = None
) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(dataset_path)

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})