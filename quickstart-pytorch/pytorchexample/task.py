"""pytorchexample: A Flower / PyTorch app."""

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import logging
from collections import defaultdict
from flwr.app import ArrayRecord, MetricRecord
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataTracker")

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

NUM_PARTITIONS = 2
IMAGES_PER_CLASS_PER_CLIENT = 10


def _resolve_dataset_dir(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path).expanduser().resolve()
    return LOCAL_DATASET_DIR


def _build_local_dataset(dataset_path: str | None) -> ImageFolder:
    dataset_dir = _resolve_dataset_dir(dataset_path)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. "
            "Set 'dataset-path' in run-config or place data in ../dataset."
        )
    return ImageFolder(root=str(dataset_dir), transform=IMAGE_TRANSFORMS)


def _get_stratified_indices(dataset: ImageFolder, partition_id: int, num_partitions: int):
    """
    Bagi rata SELURUH isi dataset ke tiap client secara stratified.
    """
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = []
    class_names = dataset.classes

    # --- TRACKING UI ---
    print(f"\n{'='*65}")
    print(f"🔍 TRACKING LOAD DATA - CLIENT {partition_id}")
    print(f"{'='*65}")

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total_class_images = len(indices)

        # Hitung jatah gambar per client untuk kelas ini
        images_per_client = total_class_images // num_partitions

        start = partition_id * images_per_client
        
        # Client terakhir ngambil semua sisa data biar gak ada image yang mubazir
        if partition_id == num_partitions - 1:
            end = total_class_images
        else:
            end = start + images_per_client

        slice_indices = indices[start:end]
        client_indices.extend(slice_indices)
        class_name = class_names[label]

        # Print tracking per folder
        print(f"📁 Kelas {class_name:<25} : dapet {len(slice_indices)} gambar (Index: {start} s/d {end-1})")

    print(f"{'-'*65}")
    print(f"✅ TOTAL GAMBAR CLIENT {partition_id} : {len(client_indices)} gambar")
    print(f"{'='*65}\n")

    return client_indices


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_path: str | None = None):
    """Load client partition dengan data yang sudah terbagi rata."""
    dataset = _build_local_dataset(dataset_path)

    # Panggil fungsi yang baru tanpa hardcode IMAGES_PER_CLASS
    indices = _get_stratified_indices(dataset, partition_id, num_partitions)

    client_subset = Subset(dataset, indices)

    # 80% train, 20% val per client
    train_size = int(0.8 * len(client_subset))
    val_size = len(client_subset) - train_size
    generator = torch.Generator().manual_seed(42 + partition_id)
    train_subset, val_subset = random_split(client_subset, [train_size, val_size], generator=generator)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset(dataset_path: str | None = None):
    dataset = _build_local_dataset(dataset_path)
    from collections import defaultdict

    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    test_indices = []
    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        start = NUM_PARTITIONS * IMAGES_PER_CLASS_PER_CLIENT
        test_indices.extend(indices[start:])

    print(f"[Centralized] Total test samples: {len(test_indices)}")
    test_subset = Subset(dataset, test_indices)
    return DataLoader(test_subset, batch_size=128, shuffle=False)


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
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    running_loss = 0.0
    
    for epoch in range(epochs):
        # Bungkus trainloader pakai tqdm biar jadi loading bar
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for batch in progress_bar:
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
            
            # Update teks di samping loading bar biar kelihatan nilai loss-nya turun/nggak
            progress_bar.set_postfix({'loss': loss.item()})
            
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