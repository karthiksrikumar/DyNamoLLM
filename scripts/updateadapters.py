# scripts/update_adapters.py
# Links:
# - TimeBench: https://github.com/zchuz/TimeBench (source for new temporal data)
# - CausalQA: https://arxiv.org/abs/2406.08642 (source for new causal data)

import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaTokenizer
from datasets import load_dataset, concatenate_datasets
from dynamo import Dynamo
from time2vec import Time2Vec
from causal_gnn import CausalGNN
from regularization import contrastive_loss

def update_adapters(config):
    """Update DYNAMO adapters with new TimeBench/CausalQA data, keeping base model frozen."""
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
    model.load_state_dict(torch.load(config['checkpoint_path']))
    model.to(device)

    # Freeze base model, only update adapters and Time2Vec
    for param in model.llama_model.parameters():
        param.requires_grad = False
    trainable_params = (
        list(model.model.time2vec.parameters()) +
        [p for layer in model.model.layers for p in layer.adapter_attn.parameters()] +
        [p for layer in model.model.layers for p in layer.adapter_ff.parameters()]
    )
    for param in trainable_params:
        param.requires_grad = True

    # Load new data (e.g., TimeBench t1/t2 split and CausalQA)
    timebench = load_dataset("timebench")
    new_time_data = timebench['train'].filter(lambda x: x["timestamp"] >= "2022-01-01")
    causal_data = load_dataset("causalqa")
    mixed_dataset = concatenate_datasets([new_time_data, causal_data['train']])
    train_loader = DataLoader(mixed_dataset, batch_size=config['batch_size'], shuffle=True)

    # Optimizer for trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=config['lr'])

    # Adaptation loop
    model.train()
    for epoch in range(config['update_epochs']):
        total_loss = 0
        for batch in train_loader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            t = torch.tensor([float(batch['timestamp'][0].split('-')[0] if batch['timestamp'][0] else 0.0)] * len(batch['text'])).to(device)
            outputs = model(**inputs, t=t)
            loss = outputs.loss

            # Add contrastive regularization
            if config['use_contrastive']:
                h_i = outputs.logits
                h_j = h_i  # Placeholder: sample from another time point
                contrastive_reg = contrastive_loss(h_i, h_j, tau=config['tau'])
                loss += config['lambda_contrastive'] * contrastive_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Update Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # Save updated model
    torch.save(model.state_dict(), config['updated_checkpoint_path'])

if __name__ == "__main__":
    config = {
        'seed': 42,
        'adapter_size': 64,
        'dim_time': 64,
        'checkpoint_path': 'checkpoints/dynamo_epoch_9.pt',
        'batch_size': 8,
        'update_epochs': 3,
        'lr': 5e-5,
        'use_contrastive': True,
        'lambda_contrastive': 0.1,
        'tau': 0.5,
        'updated_checkpoint_path': 'checkpoints/updated_dynamo.pt'
    }
    update_adapters(config)
