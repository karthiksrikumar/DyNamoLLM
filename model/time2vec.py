# time2vec.py
import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """Time2Vec embedding module for representing time t with learned parameters ω and φ.
    
    For a specified number of frequencies m, the embedding dimension is 3m, where each frequency k
    contributes three components: [ω_k t + φ_k, sin(ω_k t + φ_k), cos(ω_k t + φ_k)].
    """
    def __init__(self, m):
        """
        Initialize the Time2Vec module.

        Args:
            m (int): Number of frequencies, determining output dimension as 3*m.
        """
        super().__init__()
        self.m = m
        # Learnable parameters: ω (frequencies) and φ (phase shifts), both of size m
        self.omega = nn.Parameter(torch.randn(m))
        self.phi = nn.Parameter(torch.randn(m))

    def forward(self, t):
        """
        Compute the Time2Vec embedding for input time t.

        Args:
            t (torch.Tensor): Time input, either a scalar or tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Embedding of shape (batch_size, 3*m) or (1, 3*m) if t is scalar.
        """
        # Handle scalar input by adding a batch dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # Compute terms: broadcasting t across m frequencies
        omega_t = self.omega * t.unsqueeze(-1)  # Shape: (batch_size, m)
        phi = self.phi.unsqueeze(0)             # Shape: (1, m)
        # Linear term: ω_k t + φ_k
        linear = omega_t + phi                  # Shape: (batch_size, m)
        # Periodic terms: sin(ω_k t + φ_k) and cos(ω_k t + φ_k)
        sin_term = torch.sin(omega_t + phi)     # Shape: (batch_size, m)
        cos_term = torch.cos(omega_t + phi)     # Shape: (batch_size, m)
        # Concatenate all components along the last dimension
        embedding = torch.cat([linear, sin_term, cos_term], dim=-1)  # Shape: (batch_size, 3*m)
        return embedding

# Example usage:
# model = Time2Vec(m=2)  # Embedding dimension = 6
# t = torch.tensor([1.0, 2.0])
# embedding = model(t)  # Output shape: (2, 6)
