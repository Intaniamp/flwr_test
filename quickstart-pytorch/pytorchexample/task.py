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
    """Vision Transformer 100% disamakan dengan kode temen (model.py)"""
    def __init__(self):
        super().__init__()
        
        config = {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 128,
            "attention_heads": 4,
            "mlp_nodes": 128,
            "transformer_blocks": 4,
            "num_channels": 3,
            "num_classes": 4,
        }

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.embed_dim = config["embed_dim"]

        patch_num = (self.img_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=config["num_channels"],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.position_embedding = nn.Parameter(torch.randn(1, patch_num + 1, self.embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["attention_heads"],
            dim_feedforward=config["mlp_nodes"],
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["transformer_blocks"])
        self.mlp_head = nn.Linear(self.embed_dim, config["num_classes"])

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  
        x = x.flatten(2).transpose(1, 2)  

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.position_embedding
        x = self.transformer(x)

        x = x[:, 0]  
        x = self.mlp_head(x)

        return x

IMAGE_TRANSFORMS = Compose(
    [Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# pembagi ke klien, tiap client dapat gambar dari semua kelas (stratified)
def _get_stratified_indices(dataset: ImageFolder, partition_id: int, num_partitions: int):
    """Bagi rata isi folder TRAIN ke tiap client."""
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = []
    class_names = dataset.classes

    print(f"\n{'='*70}")
    print(f"🔍 TRACKING LOAD DATA - CLIENT {partition_id} (DARI FOLDER TRAIN)")
    print(f"{'='*70}")

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total_class_images = len(indices)

        images_per_client = total_class_images // num_partitions
        
        # Titik awal (start) dan titik akhir (end) pengambilan gambar
        start = partition_id * images_per_client
        
        if partition_id == num_partitions - 1:
            end = total_class_images
        else:
            end = start + images_per_client

        slice_indices = indices[start:end]
        client_indices.extend(slice_indices)
        
        print(f"📁 Kelas {class_names[label]:<25} : dapet {len(slice_indices):>4} gambar (Index: {start:>4} s/d {end-1:>4})")

    print(f"{'-'*70}")
    print(f"✅ TOTAL GAMBAR BELAJAR CLIENT {partition_id} : {len(client_indices)} gambar")
    print(f"{'='*70}\n")

    return client_indices


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_path: str):
    """Load data Client (HANYA DARI FOLDER TRAIN)"""
    
    # Bikin path dinamis berdasarkan config pyproject.toml
    train_dir = Path(dataset_path) / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Folder TRAIN tidak ditemukan di: {train_dir}! Tolong jalankan split_dataset.py dulu.")
        
    dataset = ImageFolder(root=str(train_dir), transform=IMAGE_TRANSFORMS)
    
    indices = _get_stratified_indices(dataset, partition_id, num_partitions)
    client_subset = Subset(dataset, indices)
    
    train_size = int(0.8 * len(client_subset))
    val_size = len(client_subset) - train_size
    generator = torch.Generator().manual_seed(42 + partition_id)
    train_subset, val_subset = random_split(client_subset, [train_size, val_size], generator=generator)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset(dataset_path: str):
    """Load data Ujian Server (HANYA DARI FOLDER VAL)"""
    
    # Bikin path dinamis berdasarkan config pyproject.toml
    val_dir = Path(dataset_path) / "val"
    
    if not val_dir.exists():
        raise FileNotFoundError(f"Folder VAL tidak ditemukan di: {val_dir}! Tolong jalankan split_dataset.py dulu.")
        
    dataset = ImageFolder(root=str(val_dir), transform=IMAGE_TRANSFORMS)
    
    print(f"\n[Centralized] 🎯 Total soal ujian murni untuk Server: {len(dataset)} gambar\n")
    return DataLoader(dataset, batch_size=128, shuffle=False)


# traing dan testing tetap di task.py biar bisa dipanggil dari server_app.py
def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["img"], batch["label"]
    images, labels = batch
    return images, labels


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0, global_params: list[torch.Tensor] | None = None):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    
    running_loss = 0.0
    correct = 0
    total = 0 

    for epoch in range(epochs):
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for batch in progress_bar:
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            if proximal_mu > 0.0 and global_params is not None:
                proximal_term = torch.tensor(0.0, device=device)
                for local_weights, global_weights in zip(net.parameters(), global_params):
                    proximal_term += torch.sum((local_weights - global_weights) ** 2)
                loss = loss + (proximal_mu / 2.0) * proximal_term
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Hitung Akurasi Train
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
    avg_trainloss = running_loss / (epochs * len(trainloader))
    train_accuracy = correct / total
    
    return avg_trainloss, train_accuracy


def test(net, testloader, device):
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


def global_evaluate(server_round: int, arrays: ArrayRecord, dataset_path: str | None = None) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset(dataset_path)
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})