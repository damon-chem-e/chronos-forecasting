import torch
from scipy.stats import norm

class DistLS(torch.nn.Module):
    """Distributional Label Smoothing centered on true values."""
    
    def __init__(self, boundaries: torch.Tensor, variance: float, special_tokens: list):
        """Create a DistLS object.

        Args:
            boundaries (torch.Tensor): Boundaries between each bin, excluding any bins for special tokens.
            variance (float): Variance.
            special_tokens (list): List of ids for bins that represent special tokens. 
                For example, [0, 1, -100] which represents PAD, EOS (end of sequence), and attention mask tokens.
        """
        super(DistLS, self).__init__()
        self.variance = variance
        self.special_tokens = special_tokens
        
        boundaries = torch.tensor(boundaries, dtype=torch.float32)
        self.boundaries = torch.cat([ # Bins for special tokens are added to the left.
            torch.arange(boundaries[0] - len(special_tokens), boundaries[0]),
            boundaries
        ])
        self.bin_edges = list(zip(self.boundaries[:-1], self.boundaries[1:]))

    def precompute_probs(self, labels: torch.Tensor) -> torch.Tensor:
        """Precompute probabilities for each class. Transforms labels of shape 
        [], [N], or [N, d_1, d_2, ..., d_K] to shape [C], [N, C], or [N, C, d_1, d_2, ..., d_K] respectively.
        N = batch size, C = num classes, d = arbitrary dimension.

        Args:
            labels (torch.Tensor): Labels including special tokens (ex. PAD, EOS) 
                and quantized and in shape [], [N], or [N, d_1, d_2, ..., d_K].

        Returns:
            torch.Tensor: Class probabilities including special tokens 
                in shape [C], [N, C], or [N, C, d_1, d_2, ..., d_K]
        """
        # Flatten labels to handle all cases uniformly
        flat_labels = labels.flatten()

        # Create mask for special tokens.
        pad_mask = torch.zeros_like(flat_labels, dtype=torch.bool)
        pad_indices = []
        for i, pad_token in enumerate(self.special_tokens):
            is_pad = (flat_labels == pad_token)
            pad_mask |= is_pad
            pad_indices.append((is_pad, i))

        # Use normal distribution around non-pad tokens
        non_pad_labels = flat_labels[~pad_mask]
        if len(non_pad_labels) > 0:
            cdf_upper = norm.cdf(self.boundaries[1:], loc=non_pad_labels[:, None], scale=self.variance)
            cdf_lower = norm.cdf(self.boundaries[:-1], loc=non_pad_labels[:, None], scale=self.variance)
            probs = torch.zeros((len(flat_labels), len(self.bin_edges)))
            probs[~pad_mask] = torch.tensor(cdf_upper - cdf_lower, dtype=torch.float32)

        # Use degenerate distribution around pad tokens
        for is_pad, pad_idx in pad_indices:
            probs[is_pad, pad_idx] = 1.0

        # Reshape the result to match the expected output shape
        result_shape = (*labels.shape, len(self.bin_edges))
        result = probs.view(result_shape)
        result = result.permute(0, -1, *range(1, result.ndim-1))  # Move new dimension C to be the 2nd dimension

        return result
    