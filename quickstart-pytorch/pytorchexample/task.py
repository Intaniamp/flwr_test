"""pytorchexample: A Flower / PyTorch app."""

import os
import collections
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import logging
from collections import defaultdict, Counter
from flwr.app import ArrayRecord, MetricRecord
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomRotation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataTracker")

class Net(nn.Module):
    """Vision Transformer (model.py)"""
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

TRAIN_TRANSFORMS = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15),          
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

VAL_TRANSFORMS = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def _get_stratified_indices(dataset: ImageFolder, partition_id: int, num_partitions: int):
    """Bagi rata isi folder TRAIN ke tiap client."""
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    client_indices = []
    class_names = dataset.classes

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        total_class_images = len(indices)
        images_per_client = total_class_images // num_partitions
        start = partition_id * images_per_client
        if partition_id == num_partitions - 1:
            end = total_class_images
        else:
            end = start + images_per_client
        slice_indices = indices[start:end]
        client_indices.extend(slice_indices)
        
    return client_indices


def print_dataset_summary(client_id, dataset_name, dataset_obj, is_subset=False):
    """Fungsi untuk menampilkan ringkasan data yang diload secara rapi."""
    print(f"\n" + "="*55)
    print(f"📦 [Client {client_id}] Memuat Data: {dataset_name}")
    
    if is_subset:
        client_subset = dataset_obj.dataset
        base_image_folder = client_subset.dataset
        
        classes = base_image_folder.classes
        
        # Ambil label dengan melacak index dari 2 lapis Subset tersebut
        targets = [base_image_folder.targets[client_subset.indices[i]] for i in dataset_obj.indices]
    else:
        # Untuk Real Data yang langsung berasal dari ImageFolder
        classes = dataset_obj.classes
        targets = dataset_obj.targets

    class_counts = collections.Counter(targets)
    
    print("📊 Distribusi Kelas:")
    for class_idx, count in class_counts.items():
        print(f"   - {classes[class_idx]}: {count} gambar")
    print("="*55 + "\n")


def load_data(partition_id: int, num_partitions: int, batch_size: int, kaggle_path: str, real_path: str):
    """Load data Client (Kaggle atau Real Data) dengan Log Transparan"""
    
    if partition_id == 2:
        real_train_dir = Path(real_path) / "train"
        real_val_dir = Path(real_path) / "val"
        
        if not real_train_dir.exists():
            raise FileNotFoundError(f"Folder real data tidak ditemukan: {real_train_dir}")
            
        train_subset = ImageFolder(root=str(real_train_dir), transform=TRAIN_TRANSFORMS)
        val_subset = ImageFolder(root=str(real_val_dir), transform=VAL_TRANSFORMS)
        
        print_dataset_summary(partition_id, f"Real Data ({real_train_dir})", train_subset)
        
    else:
        kaggle_train_dir = Path(kaggle_path) / "train"
        
        if not kaggle_train_dir.exists():
            raise FileNotFoundError(f"Folder TRAIN tidak ditemukan di: {kaggle_train_dir}")
            
        dataset = ImageFolder(root=str(kaggle_train_dir), transform=TRAIN_TRANSFORMS)
        
        kaggle_partitions = num_partitions - 1
        indices = _get_stratified_indices(dataset, partition_id, kaggle_partitions)
        client_subset = Subset(dataset, indices)
        
        train_size = int(0.8 * len(client_subset))
        val_size = len(client_subset) - train_size
        generator = torch.Generator().manual_seed(42 + partition_id)
        
        train_subset, val_subset = random_split(client_subset, [train_size, val_size], generator=generator)

        print_dataset_summary(partition_id, f"Kaggle Data ({kaggle_train_dir})", train_subset, is_subset=True)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset(kaggle_path: str, real_path: str):
    """Load data Ujian Server (GABUNGAN KAGGLE + REAL DATA)"""
    
    kaggle_val_dir = Path(kaggle_path) / "val"
    real_val_dir = Path(real_path) / "val" 
    
    if not kaggle_val_dir.exists():
        raise FileNotFoundError(f"Folder VAL Kaggle tidak ditemukan di: {kaggle_val_dir}")
    if not real_val_dir.exists():
        raise FileNotFoundError(f"Folder VAL Real Data tidak ditemukan di: {real_val_dir}")
        
    kaggle_dataset = ImageFolder(root=str(kaggle_val_dir), transform=VAL_TRANSFORMS)
    real_dataset = ImageFolder(root=str(real_val_dir), transform=VAL_TRANSFORMS)
    
    combined_dataset = ConcatDataset([kaggle_dataset, real_dataset])
    
    print(f"\n[Centralized] 🎯 Total soal ujian murni untuk Server: {len(combined_dataset)} gambar")
    print(f"    - Dari Kaggle: {len(kaggle_dataset)} gambar")
    print(f"    - Dari Real Data: {len(real_dataset)} gambar\n")
    
    return DataLoader(combined_dataset, batch_size=128, shuffle=False)


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["img"], batch["label"]
    images, labels = batch
    return images, labels


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0, global_params: list[torch.Tensor] | None = None):
    net.to(device)
    
    if isinstance(trainloader.dataset, Subset):
        base_dataset = trainloader.dataset.dataset.dataset
        client_indices = trainloader.dataset.dataset.indices
        train_indices = trainloader.dataset.indices
        subset_targets = [base_dataset.targets[client_indices[i]] for i in train_indices]
    else:
        subset_targets = trainloader.dataset.targets
    
    class_counts = Counter(subset_targets)
    counts = [class_counts.get(i, 1) for i in range(4)] 
    
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights / weights.sum()
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    
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


def global_evaluate(server_round: int, arrays: ArrayRecord, kaggle_path: str, real_path: str) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset(kaggle_path, real_path)
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})