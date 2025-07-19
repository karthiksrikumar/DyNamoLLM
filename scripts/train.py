# scripts/train.py
# Links:
# - TimeBench: https://github.com/zchuz/TimeBench (evaluation benchmark, not used for training)
# - CausalQA: https://arxiv.org/abs/2406.08642 (causal evaluation benchmark)
# - TempReason: https://arxiv.org/abs/2402.10156 (potential training data source)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaConfig, LlamaTokenizer
from datasets import load_dataset
from dynamo import Dynamo
from time2vec import Time2Vec
from causal_gnn import CausalGNN
from regularization import contrastive_loss

def train(rank, world_size, config):
    """Train DYNAMO on a temporal dataset with multi-GPU support, preparing for TimeBench evaluation.
    
    Only adapters and Time2Vec parameters are trained, keeping LLaMA-7B frozen for efficiency.
    """
    # Initialize distributed training
    if world_size > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://', world_size=world_size, rank=rank
        )
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])

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
    model = Dynamo(llama_config, adapter_size=config['adapter_size'], dim_time=config['dim_time'])
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Freeze base LLaMA parameters
    for param in model.llama_model.parameters():
        param.requires_grad = False
    trainable_params = (
        list(model.model.time2vec.parameters()) +
        [p for layer in model.model.layers for p in layer.adapter_attn.parameters()] +
        [p for layer in model.model.layers for p in layer.adapter_ff.parameters()]
    )

    # Load training dataset (e.g., TempReason or CommonCrawl subset, not TimeBench test)
    dataset = load_dataset(config['dataset_name'])
    train_loader = DataLoader(dataset['train'], batch_size=config['batch_size'], shuffle=True)

    # Optimizer for trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=config['lr'])

    # Training loop
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            t = torch.tensor([float(batch['timestamp'][0].split('-')[0])] * len(batch['text'])).to(device)
            outputs = model(**inputs, t=t)
            loss = outputs.loss

            # Add causal invariance regularization
            if config['use_contrastive']:
                h_i = outputs.logits
                h_j = h_i  # Placeholder: sample from another time point (requires causal data)
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
        'seed': 42,
        'adapter_size': 64,
        'dim_time': 64,
        'dataset_name': 'tempreason',  # Use TempReason or CommonCrawl subset
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
