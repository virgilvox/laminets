# FieldPoint and LaminaField

import torch
import torch.nn as nn
from laminet_micro.forces import semantic_attraction

class FieldPoint(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.position = nn.Parameter(torch.randn(embed_dim))
        self.velocity = nn.Parameter(torch.zeros(embed_dim))
        self.mass = nn.Parameter(torch.randn(1))
        self.charge = nn.Parameter(torch.randn(1))
        self.decay_rate = nn.Parameter(torch.randn(1))

class LaminaField(nn.Module):
    def __init__(self, input_embeddings):
        super().__init__()
        self.points = nn.ModuleList([FieldPoint(embed_dim=input_embeddings.shape[-1]) for _ in range(len(input_embeddings))])
        self.embed_points(input_embeddings)

    def embed_points(self, embeddings):
        for point, embed in zip(self.points, embeddings):
            point.position.data = embed

    def compute_net_force(self, point_idx):
        forces = []
        for j, other_point in enumerate(self.points):
            if j == point_idx:
                continue
            forces.append(semantic_attraction(self.points[point_idx], other_point))
        return torch.sum(torch.stack(forces), dim=0)

    def evolve(self, dt=0.01):
        for idx, point in enumerate(self.points):
            net_force = self.compute_net_force(idx)
            point.velocity = point.velocity + net_force * dt
            point.position = point.position + point.velocity * dt
            point.velocity *= (1.0 - point.decay_rate.abs() * dt)

    def measure_potential_energy(self):
        energy = 0.0
        for point in self.points:
            energy += point.velocity.norm(p=2) ** 2
        return energy
