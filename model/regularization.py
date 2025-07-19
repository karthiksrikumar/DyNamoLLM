# regularization.py
import torch
import torch.nn.functional as F

def contrastive_loss(h_i, h_j, tau=0.5, negative_samples=None):
    """
    Contrastive loss approximating D_JS(P_Ψ^(t_i) || P_Ψ^(t_j)) for causal invariance.

    Encourages representations at times t_i and t_j to be similar if causally invariant,
    and dissimilar otherwise, using cosine similarity.

    Args:
        h_i (torch.Tensor): Representation/prediction at time t_i, shape (batch_size, dim).
        h_j (torch.Tensor): Representation/prediction at time t_j, shape (batch_size, dim).
        tau (float): Temperature parameter for scaling similarity.
        negative_samples (torch.Tensor, optional): Negative samples, shape (batch_size, num_neg, dim).

    Returns:
        torch.Tensor: Scalar contrastive loss value.
    """
    # Compute cosine similarity for positive pairs (h_i, h_j)
    sim_pos = F.cosine_similarity(h_i, h_j, dim=-1) / tau  # Shape: (batch_size,)
    if negative_samples is not None:
        # Compute similarity with negative samples
        sim_neg = F.cosine_similarity(h_i.unsqueeze(1), negative_samples, dim=-1) / tau  # Shape: (batch_size, num_neg)
        sim_all = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # Shape: (batch_size, 1 + num_neg)
    else:
        sim_all = sim_pos.unsqueeze(1)  # Shape: (batch_size, 1)
    # Compute contrastive loss using log-softmax
    log_probs = F.log_softmax(sim_all, dim=1)
    loss = -log_probs[:, 0].mean()  # Negative log probability of positive pair
    return loss

# Theoretical Drift Bound Documentation
"""
Theoretical Guarantee: Output Drift Bound

Under the assumption of L-Lipschitz continuity of the adapter parameters with respect to time t,
the output drift of the model ℳ(x,t) is bounded as follows:

||ℳ(x,t) - ℳ(x,t+Δ)||_2 ≤ L * κ * |Δ| * ||φ'(t)||_2

Where:
- L is the Lipschitz constant of the adapter parameters.
- κ = ||𝐖_o||_2 * ||�{W}_t||_2 is the condition number, with 𝐖_o and 𝐖_t being the output and
  temporal projection matrices, respectively.
- φ'(t) is the derivative of the temporal embedding with respect to time.
- Δ is the time difference.

This bound ensures that the model's output changes smoothly over time, providing stability
in temporal adaptation. The proof relies on the chain rule and Lipschitz properties of the
adapter modulation Δ 𝐇^ℓ = 𝐖_o σ(�{W}_t φ(t) + 𝐖_g f_g(𝒢_t)), assuming bounded gradients
of φ(t) and f_g(𝒢_t).
"""

# Example usage:
# h_i = torch.randn(4, 10)  # Representations at t_i
# h_j = torch.randn(4, 10)  # Representations at t_j
# neg = torch.randn(4, 3, 10)  # Negative samples
# loss = contrastive_loss(h_i, h_j, tau=0.5, negative_samples=neg)
