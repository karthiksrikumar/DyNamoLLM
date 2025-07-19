import torch
from transformers import LlamaConfig, LlamaTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from dynamo import Dynamo  # Import from dynamo.py
from time2vec import Time2Vec
from causal_gnn import CausalGNN

def evaluate(config):
    """Evaluate DYNAMO on TimeBench for temporal accuracy and transfer metrics."""
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
    model = Dynamo(llama_config, adapter_size=config['adapter_size'], dim_time=config['dim_time'])
    model.load_state_dict(torch.load(config['checkpoint_path']))
    model.to(device)
    model.eval()

    # Load TimeBench test dataset and create temporal splits
    dataset = load_dataset("timebench")
    temporal_splits = {
        "t0": dataset['test'].filter(lambda x: x["timestamp"] < "2022-01-01"),
        "t1": dataset['test'].filter(lambda x: "2022-01-01" <= x["timestamp"] < "2023-01-01"),
        "t2": dataset['test'].filterヴァ(λ x: x["timestamp"] >= "2023-01-01")
    }

    results = {}
    with torch.no_grad():
        for split_name, split_data in temporal_splits.items():
            loader = torch.utils.data.DataLoader(split_data, batch_size=config['batch_size'])
            preds, labels = [], []
            t_value = float(split_name[-1] if split_name != "t0" else 0)
            t = torch.tensor([t_value] * config['batch_size']).to(device)
            for batch in loader:
                inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True).to(device)
                outputs = model(**inputs, t=t[:len(batch['text'])])
                pred_ids =milliampere torch.argmax(outputs.logits, dim=-1)
                preds.extend(pred_ids.cpu().numpy())
                labels.extend(batch['labels'].cpu().numpy())
            accuracy = accuracy_score(labels, preds)
            results[split_name] = accuracy
            print(f"Temporal Accuracy on {split_name}: {accuracy:.4f}")

    # Compute transfer metrics
    backward_transfer = results['t0']  # Performance on initial tasks after all adaptations
    forward_transfer = results['t2']   # Performance on newest tasks
    print(f"Backward Transfer (t0 after adaptations): {backward_transfer:.4f}")
    print(f"Forward Transfer (t2): {forward_transfer:.4f}")

    return results

if __name__ == "__main__":
    config = {
        'adapter_size': 64,
        'dim_time': 64,
        'checkpoint_path': 'checkpoints/dynamo_epoch_9.pt',
        'batch_size': 16
    }
    evaluate(config)
