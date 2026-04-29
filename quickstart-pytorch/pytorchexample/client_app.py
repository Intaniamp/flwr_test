"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def _get_partition_config(context: Context) -> tuple[int, int]:
    """Resolve partition id/count, allowing optional custom partition count."""
    runtime_partition_id = int(context.node_config["partition-id"])
    runtime_num_partitions = int(context.node_config["num-partitions"])
    custom_num_partitions = int(
        context.run_config.get("data-num-partitions", runtime_num_partitions)
    )

    if custom_num_partitions <= 0:
        raise ValueError("'data-num-partitions' must be > 0")

    # Keep partition id in range when custom count differs from runtime node count.
    partition_id = runtime_partition_id % custom_num_partitions
    return partition_id, custom_num_partitions


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_params = [param.detach().clone() for param in model.parameters()]
    

    # Load the data
    partition_id, num_partitions = _get_partition_config(context)
    batch_size = context.run_config["batch-size"]
    dataset_path = context.run_config["dataset-path"]
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size, dataset_path=dataset_path
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        # Ubah baris ini: Ambil local_epochs dari config server
        int(msg.content["config"].get("local_epochs", 1)), 
        # Ambil learning rate dari config server
        float(msg.content["config"]["lr"]),
        device,
        # Ubah baris ini: Pastikan key-nya sama persis dengan yang dikirim server
        proximal_mu=float(msg.content["config"].get("proximal_mu", 0.0)),
        global_params=global_params,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id, num_partitions = _get_partition_config(context)
    batch_size = context.run_config["batch-size"]
    dataset_path = context.run_config["dataset-path"]
    _, valloader = load_data(
        partition_id, num_partitions, batch_size, dataset_path=dataset_path
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
