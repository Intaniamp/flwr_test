"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from pytorchexample.custom_strategy import CustomFedProx
from pytorchexample.task import Net, global_evaluate

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config (pastikan typo fraction-train sudah diperbaiki)
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    fraction_train: float = context.run_config["fraction-train"]
    dataset_path: str = context.run_config["dataset-path"]
    proximal_mu: float = context.run_config["proximal-mu"]
    
    # Tambahkan variabel untuk membaca local epochs dari config (default ke 3 jika tidak ada)
    local_epochs: int = context.run_config.get("local-epochs", 1)

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    best_accuracy = 0.0

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
