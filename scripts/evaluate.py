# scripts/evaluate.py
# Links:
# - TimeBench: https://github.com/zchuz/TimeBench (primary temporal reasoning benchmark)
# - CausalQA: https://arxiv.org/abs/2406.08642 (causal reasoning benchmark)

import torch
from transformers import LlamaConfig, LlamaTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score
from dynamo import Dynamo
from time2vec import Time2Vec
from causal_gnn import CausalGNN

def evaluate(config):
    """Evaluate DYNAMO on TimeBench and CausalQA for temporal and causal performance."""
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
    model.eval()

    # Load TimeBench test dataset and create temporal splits
    timebench = load_dataset("timebench")
    temporal_splits = {
        "t0": timebench['test'].filter(lambda x: x["timestamp"] < "2022-01-01"),
        "t1": timebench['test'].filter(lambda x: "2022-01-01" <= x["timestamp"] < "2023-01-01"),
        "t2": timebench['test'].filter(lambda x: x["timestamp"] >= "2023-01-01")
    }

    # Evaluate on TimeBench
    timebench_results = {}
    with torch.no_grad():
        for split_name, split_data in temporal_splits.items():
            loader = torch.utils.data.DataLoader(split_data, batch_size=config['batch_size'])
            preds, labels = [], []
            t_value = float(split_name[-1] if split_name != "t0" else 0)
            t = torch.tensor([t_value] * config['batch_size']).to(device)
            for batch in loader:
                inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
                t_batch = t[:len(batch['text'])]
                outputs = model(**inputs, t=t_batch)
                pred_ids = torch.argmax(outputs.logits, dim=-1)
                preds.extend(pred_ids.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
            accuracy = accuracy_score(labels, preds)
            timebench_results[split_name] = accuracy
            print(f"Temporal Accuracy on TimeBench {split_name}: {accuracy:.4f}")

    # Compute transfer metrics
    backward_transfer = timebench_results['t0']
    forward_transfer = timebench_results['t2']
    print(f"Backward Transfer (t0 after adaptations): {backward_transfer:.4f}")
    print(f"Forward Transfer (t2): {forward_transfer:.4f}")

    # Load CausalQA test dataset
    causalqa = load_dataset("causalqa")
    causal_loader = torch.utils.data.DataLoader(causalqa['test'], batch_size=config['batch_size'])
    causal_preds, causal_labels = [], []
    with torch.no_grad():
        for batch in causal_loader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
            t = torch.tensor([0.0] * len(batch['text'])).to(device)  # No timestamp for causal tasks
            outputs = model(**inputs, t=t)
            preds = torch.argmax(outputs.logits, dim=-1)
            causal_preds.extend(preds.cpu().numpy())
            causal_labels.extend(batch['labels'].cpu().numpy())
    causal_f1 = f1_score(causal_labels, causal_preds, average='macro')
    print(f"Causal F1 Score on CausalQA: {causal_f1:.4f}")

    return {
        'timebench': timebench_results,
        'backward_transfer': backward_transfer,
        'forward_transfer': forward_transfer,
        'causal_f1': causal_f1
    }

if __name__ == "__main__":
    config = {
        'seed': 42,
        'adapter_size': 64,
        'dim_time': 64,
        'checkpoint_path': 'checkpoints/dynamo_epoch_9.pt',
        'batch_size': 16
    }
    evaluate(config)
