# LaminetMicro: Simple field-evolution model

import torch
import torch.nn as nn
from laminet_micro.field import LaminaField

class LaminetMicro(nn.Module):
    def __init__(self, embed_dim=64, decoder_out_dim=6):
        super().__init__()
        self.embedder = nn.Linear(128, embed_dim)
        self.decoder = nn.Linear(embed_dim, decoder_out_dim)

    def forward(self, x, time_steps=50, dt=0.01):
        embeddings = self.embedder(x)
        field = LaminaField(embeddings)

        for _ in range(time_steps):
            field.evolve(dt)

        final_positions = torch.stack([p.position for p in field.points])
        output = self.decoder(final_positions.mean(dim=0))
        potential_energy = field.measure_potential_energy()
        return output, potential_energy
