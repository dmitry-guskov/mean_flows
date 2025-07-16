import torch
import numpy as np



class DistributionGenerator:
    def generate(self, num_points):
        raise NotImplementedError


class GaussianGenerator(DistributionGenerator):
    def __init__(self, n_dims=2, noise_std=1.0):
        self.n_dims = n_dims
        self.noise_std = noise_std

    def generate(self, num_points):
        return torch.randn(num_points, self.n_dims) * self.noise_std


class SpiralGenerator(DistributionGenerator):
    def __init__(self, noise_std=0.1, n_turns=2, radius_scale=2):
        self.noise_std = noise_std
        self.n_turns = n_turns
        self.radius_scale = radius_scale

    def generate(self, num_points):
        max_angle = 2 * np.pi * self.n_turns
        t = torch.linspace(0, max_angle, num_points)
        t = t * torch.pow(torch.rand(num_points), 0.5)

        r = self.radius_scale * (t / max_angle + 0.1)
        x = r * torch.cos(t)
        y = r * torch.sin(t)

        x += torch.randn(num_points) * self.noise_std
        y += torch.randn(num_points) * self.noise_std
        return torch.stack([x, y], dim=1)


class CheckerboardGenerator(DistributionGenerator):
    def __init__(self, grid_size=3, scale=2.0, device='cpu'):
        self.grid_size = grid_size
        self.scale = scale
        self.device = device

    def generate(self, num_points):
        grid_length = 2 * self.scale / self.grid_size
        samples = torch.zeros(0, 2).to(self.device)

        while samples.shape[0] < num_points:
            new_samples = (torch.rand(num_points, 2).to(self.device) - 0.5) * 2 * self.scale
            x_mask = torch.floor((new_samples[:, 0] + self.scale) / grid_length) % 2 == 0
            y_mask = torch.floor((new_samples[:, 1] + self.scale) / grid_length) % 2 == 0
            accept_mask = torch.logical_xor(~x_mask, y_mask)
            samples = torch.cat([samples, new_samples[accept_mask]], dim=0)

        return samples[:num_points]


class MultiGaussGenerator(DistributionGenerator):
    def __init__(self, centers=None, noise_std=0.1, device='cpu'):
        self.centers = centers or [
            [1., 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [.75, .75],
            [-.75, -.75],
        ]
        self.noise_std = noise_std
        self.device = device

    def generate(self, num_points):
        centers = torch.tensor(self.centers, dtype=torch.float32).to(self.device)
        num_centers = centers.shape[0]
        assignments = torch.randint(0, num_centers, (num_points,), device=self.device)
        noise = torch.randn(num_points, 2, device=self.device) * self.noise_std
        samples = centers[assignments] + noise
        return samples
