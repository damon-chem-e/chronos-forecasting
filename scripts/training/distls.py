import torch
from torch.utils.data import DataLoader
from scipy.stats import norm

class DistLS(torch.nn.Module):
    """Distributional Label Smoothing centered on true values."""
    
    def __init__(self, boundaries: torch.Tensor, variance: float):
        """Create a DistLS object. N = batch size, C = num classes, d = arbitrary dimension.

        Args:
            boundaries (torch.Tensor): Boundaries between each bin.
            variance (float): Variance.
        """
        super(DistLS, self).__init__()
        self.variance = variance
        self.boundaries = torch.tensor(boundaries, dtype=torch.float32)
        self.bin_edges = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        self.bin_centers = (self.boundaries[:-1] + self.boundaries[1:]) / 2 # For visualization, not used

    def precompute_probs(self, labels: torch.Tensor) -> torch.Tensor:
        """Precompute probabilities for each class. Transforms labels of shape 
        [], [N], or [N, d_1, d_2, ..., d_K] to shape [C], [N, C], or [N, C, d_1, d_2, ..., d_K] respectively.

        Args:
            labels (torch.Tensor): Quantized and in shape [], [N], or [N, d_1, d_2, ..., d_K].

        Returns:
            torch.Tensor: Class probabilities in shape [C], [N, C], or [N, C, d_1, d_2, ..., d_K]
        """
        # Flatten labels to handle all cases uniformly
        flat_labels = labels.flatten()

        # Vectorized computation of CDF differences for all labels and bin edges
        cdf_upper = norm.cdf(self.boundaries[1:], loc=flat_labels[:, None], scale=self.variance)
        cdf_lower = norm.cdf(self.boundaries[:-1], loc=flat_labels[:, None], scale=self.variance)
        probs = torch.Tensor(cdf_upper - cdf_lower)

        # Reshape the result to match the expected output shape
        result_shape = (*labels.shape, len(self.bin_edges))
        result = probs.view(result_shape)
        result = result.permute(0, -1, *range(1, result.ndim-1))  # Move new dimension C to be the 2nd dimension

        return result
    