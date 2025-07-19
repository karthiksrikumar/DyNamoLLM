import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig, LlamaModel, LlamaDecoderLayer

class Adapter(nn.Module):
    """A simple feed-forward adapter with a bottleneck structure."""
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

# Time2Vec Module for Temporal Embeddings
class Time2Vec(nn.Module):
    """Encodes time into a vector using periodic functions (sine and cosine)."""
    def __init__(self, dim_time):
        super().__init__()
        assert dim_time % 2 == 0, "dim_time must be even for sin/cos pairs"
        self.w = nn.Parameter(torch.randn(dim_time // 2))  # Frequencies
        self.phi = nn.Parameter(torch.randn(dim_time // 2))  # Phase shifts

    def forward(self, t):
        if t.dim() == 0:  # Handle scalar input
            t = t.unsqueeze(0)
        w_t = self.w * t.unsqueeze(-1)  # (batch_size, dim_time // 2)
        sin_part = torch.sin(w_t + self.phi)
        cos_part = torch.cos(w_t + self.phi)
        time_vec = torch.cat([sin_part, cos_part], dim=-1)  # (batch_size, dim_time)
        return time_vec

class TemporalAdapter(nn.Module):
    """Adapter modulated by temporal embeddings from Time2Vec."""
    def __init__(self, hidden_size, adapter_size, dim_time):
        super().__init__()
        self.adapter = Adapter(hidden_size, adapter_size)
        self.W_scale = nn.Linear(dim_time, hidden_size)  # Projects time_vec to hidden_size
        self.act = nn.Sigmoid()  # Activation for scaling factor

    def forward(self, x, time_vec):
        adapter_out = self.adapter(x)  # (batch_size, seq_len, hidden_size)
        scale = self.act(self.W_scale(time_vec)).unsqueeze(1)  # (batch_size, 1, hidden_size)
        modulated_adapter_out = adapter_out * scale  # Element-wise modulation
        return modulated_adapter_out

class DynamoDecoderLayer(LlamaDecoderLayer):
    """Extends LlamaDecoderLayer to include temporal adapters after attention and feed-forward."""
    def __init__(self, config, layer_idx, adapter_size, dim_time):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size
        self.adapter_attn = TemporalAdapter(hidden_size, adapter_size, dim_time)
        self.adapter_ff = TemporalAdapter(hidden_size, adapter_size, dim_time)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, 
                time_vec=None, **kwargs):
        # Original forward pass with residual connections
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self_attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self_attn_output
        # Add adapter after attention, modulated by time_vec
        if time_vec is not None:
            hidden_states = hidden_states + self.adapter_attn(hidden_states, time_vec)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        # Add adapter after feed-forward, modulated by time_vec
        if time_vec is not None:
            hidden_states = hidden_states + self.adapter_ff(hidden_states, time_vec)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

class DynamoModel(LlamaModel):
    def __init__(self, config, adapter_size, dim_time):
        super().__init__(config)
        self.time2vec = Time2Vec(dim_time)
        # Replace layers with DynamoDecoderLayer instances
        self.layers = nn.ModuleList([
            DynamoDecoderLayer(config, layer_idx, adapter_size, dim_time) 
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(self, *args, t=None, **kwargs):
        # Compute time_vec from input time t if provided
        time_vec = self.time2vec(t) if t is not None else None
        # Pass time_vec to layers via kwargs
        return super().forward(*args, time_vec=time_vec, **kwargs)

# Main Dynamo Model
class Dynamo(LlamaForCausalLM):
    def __init__(self, config, adapter_size=64, dim_time=64):
        """
        Args:
            config: LlamaConfig instance for LLaMA-7B.
            adapter_size: Size of the adapter bottleneck (64).
            dim_time: Dimension of the Time2Vec embedding.
        """
        super().__init__(config)
        self.model = DynamoModel(config, adapter_size, dim_time)

    def forward(self, *args, t=None, **kwargs):
        return super().forward(*args, t=t, **kwargs)

# Example usage (commented out)
if __name__ == "__main__":
    # Initialize configuration for LLaMA-7B
    config = LlamaConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_position_embeddings=2048,
    )
    model = Dynamo(config, adapter_size=64, dim_time=64)
    
    # Set only adapter and Time2Vec parameters as trainable
    trainable_params = (
        list(model.model.time2vec.parameters()) +
        [p for layer in model.model.layers for p in layer.adapter_attn.parameters()] +
        [p for layer in model.model.layers for p in layer.adapter_ff.parameters()]
    )
    for p in model.parameters():
        p.requires_grad = False
    for p in trainable_params:
        p.requires_grad = True
    
    # Example forward pass with dummy data
    input_ids = torch.randint(0, config.vocab_size, (2, 10))  # batch_size=2, seq_len=10
    t = torch.tensor([2023.0, 2024.0])  # Time periods for each batch item
    outputs = model(input_ids=input_ids, t=t)
    print(outputs.logits.shape)  # Should be (2, 10, vocab_size)
