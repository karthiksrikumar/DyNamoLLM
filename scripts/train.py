import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaTokenizer
from datasets import load_dataset
from dynamo import Dynamo  # Import from dynamo.py
from time2vec import Time2Vec
from causal_gnn import CausalGNN
from regularization import contrastive_loss

def train(rank, world_size, config):
    """Main training function for DYNAMO on TimeBench, supporting multi-GPU."""
    # Initialize distributed training if multi-GPU
    if world_size > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=world_size, rank=rank
        )
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-ai/llama-7b")

    # Initialize DYNAMO model
    llama_config = LlamaConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_position_embeddings=2048,
    )
    time2vec = Time2Vec(m=config['m'])
    causal_gnn = CausalGNN(num_nodes=config['num_nodes'], hidden_dim=config['hidden_dim'], output_dim=config['output_dim'])
    model = Dynamo(llama_config, adapter_size=config['adapter_size'], dim_time=config['dim_time'])
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Load TimeBench dataset and create temporal splits
    dataset = load_dataset("timebench")  # Replace with actual dataset path
    temporal_splits = {
        "t0": dataset['train'].filter(lambda x: x["timestamp"] < "2022-01-01"),
        "t1": dataset['train'].filter(lambda x: "2022-01-01" <= x["timestamp"] < "2023-01-01"),
        "t2": dataset['train'].filter(lambda x: x["timestamp"] >= "2023-01-01")
    }
    train_loader = DataLoader(temporal_splits['t0'], batch_size=config['batch_size'], shuffle=True)

    # Optimizer for adapters and Time2Vec parameters
    trainable_params = (
        list(model.model.time2vec.parameters()) +
        [p for layer in model.model.layers for p in layer.adapter_attn.parameters()] +
        [p for layer in model.model.layers for p in layer.adapter_ff.parameters()]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=config['lr'])

    # Training loop
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            t = torch.tensor([float(batch['timestamp'][0].split('-')[0])]).to(device)  # Year as float
            outputs = model(**inputs, t=t)
            loss = outputs.loss

            # Add causal invariance regularization
            if config['use_contrastive']:
                h_i = outputs.logits  # Use logits as representations
                h_j = h_i  # Placeholder: sample from another time point
                contrastive_reg = contrastive_loss(h_i, h_j, tau=config['tau'])
                loss += config['lambda_contrastive'] * contrastive_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoints/dynamo_epoch_{epoch}.pt")

    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    config = {
        'm': 10,  # Number of frequencies for Time2Vec
        'num_nodes': 50,
        'hidden_dim': 128,
        'output_dim': 64,
        'adapter_size': 64,
        'dim_time': 64,
        'dataset_name': 'timebench',
        'batch_size': 16,
        'epochs': 10,
        'lr': 1e-4,
        'use_contrastive': True,
        'lambda_contrastive': 0.1,
        'tau': 0.5
    }
    world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.multiprocessing.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train(0, 1, config)
