"""pytorchexample: A Flower / PyTorch app."""

import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from pytorchexample.custom_strategy import CustomFedProx
from pytorchexample.task import Net, global_evaluate

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_train: float = context.run_config["fraction-train"]
    dataset_path: str = context.run_config["dataset-path"]
    proximal_mu: float = context.run_config["proximal-mu"]
    
    local_epochs: int = context.run_config.get("local-epochs", 1)

    # Load global model dasar
    global_model = Net()
    best_accuracy = 0.0

    # FITUR AUTO-RESUME / SMART RESTART
    model_path = "best_model_padi.pt"
    if os.path.exists(model_path):
        print(f"\n[RESUME] Menemukan file {model_path}!")
        print("Mencoba memuat ingatan model sebelumnya...")
        
        try:
            # 1. Coba pakaikan bobot lama ke model
            global_model.load_state_dict(torch.load(model_path))
            
            # 2. Kalau sukses masuk (nggak error), uji coba kilat
            print("Berhasil! Mengukur akurasi awal dari model yang dimuat...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            global_model.to(device)
            
            from pytorchexample.task import load_centralized_dataset, test
            test_dataloader = load_centralized_dataset(dataset_path)
            _, initial_acc = test(global_model, test_dataloader, device)
            
            best_accuracy = initial_acc # Kunci rekor awal
            print(f"✅ Akurasi awal dikunci pada: {best_accuracy:.4f}\n")
            
        except RuntimeError as e:
            # 3. Kalau error karena arsitektur ViT-nya berubah, tangkap error-nya di sini!
            print("\n🚨 [PERINGATAN] Arsitektur ViT sepertinya berubah (Beda jumlah kelas/layer)!")
            print("🤖 Mengabaikan model lama. ViT akan mulai belajar dari 0 (Bayi)!\n")
            
            # Reset model jadi bayi baru, dan kembalikan rekor ke 0
            global_model = Net()
            best_accuracy = 0.0
            
    else:
        print("\n[INFO] File best_model_padi.pt tidak ditemukan. Model akan mulai belajar dari 0.\n")

    # Masukkan bobot (baik yang dari resume, maupun yang bayi) ke dalam ArrayRecord
    arrays = ArrayRecord(global_model.state_dict())

    # ==========================================

    def evaluate_and_save(server_round, current_arrays):
        nonlocal best_accuracy
        
        metrics_record = global_evaluate(server_round, current_arrays, dataset_path=dataset_path)
        
        loss = float(metrics_record["loss"])
        accuracy = float(metrics_record["accuracy"])
        
        if accuracy > best_accuracy:
            print(f"\n🌟 REKOR BARU! Round {server_round}: {accuracy:.4f} (Melampaui {best_accuracy:.4f})")
            best_accuracy = accuracy
            state_dict = current_arrays.to_torch_state_dict()
            torch.save(state_dict, "best_model_padi.pt")
            print("💾 Model paling jenius berhasil diamankan ke 'best_model_padi.pt'!\n")
            
        return metrics_record

    # Initialize FedProx strategy
    strategy = CustomFedProx(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        proximal_mu=proximal_mu,
    )

    # Start strategy, run FedProx for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "lr": lr,
            "local_epochs": local_epochs,
            "proximal_mu": proximal_mu  
        }),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_and_save
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")