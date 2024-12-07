import torch
import torch.nn as nn
import math


class lap_ablation_none_module(torch.nn.Module):
    def __init__(self, channels=None):
        super(lap_ablation_none_module, self).__init__()

    @staticmethod
    def get_module_name():
        return "lap_ablation_none"

    def forward(self, x):
        epsilon = 1e-8  # Small number to avoid division by zero
        b, c, h, w = x.size()
        n = w * h - 1
        # Compute |x - mu|
        x_minus_mu = x - x.mean(dim=[2, 3], keepdim=True)
        x_minus_mu_abs = torch.abs(x_minus_mu)

        # Compute the standard deviation (std) of x
        std = torch.sqrt(
            ((x_minus_mu) ** 2).mean(dim=[2, 3], keepdim=True) + epsilon)

        # Use std/sqrt(2) as the scale parameter b for the Laplacian distribution
        lap_b = std / math.sqrt(2)

        # Compute the Laplace PDF
        adjusted_lap_pdf = (torch.exp(-x_minus_mu_abs / lap_b)) / (2 * lap_b)

        return x * (1 / (adjusted_lap_pdf+1))
