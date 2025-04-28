import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader

# 1. Define FieldPoint
class FieldPoint(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.position = nn.Parameter(torch.randn(embed_dim))
        self.velocity = nn.Parameter(torch.zeros(embed_dim))
        self.mass = nn.Parameter(torch.randn(1))
        self.charge = nn.Parameter(torch.randn(1))
        self.decay_rate = nn.Parameter(torch.randn(1))

# 2. Define LaminaField
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
            direction = other_point.position - self.points[point_idx].position
            distance = direction.norm(p=2) + 1e-6
            semantic_force = (self.points[point_idx].charge * other_point.charge) / (distance**2)
            forces.append(semantic_force * direction / distance)
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

# 3. Define Laminet Model
class LaminetMicro(nn.Module):
    def __init__(self, embed_dim=64, decoder_out_dim=6):  # 6 classes
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

# 4. Define Dataset
class LaminetDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        self.data = data
        self.label_map = {theme: idx for idx, theme in enumerate(sorted(set(ex["label"] for ex in data)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = torch.randn(len(example["text"].split()), 128)  # Fake embeddings for now
        label = self.label_map[example["label"]]
        return tokens, torch.tensor(label)

# 5. Train
def train():
    dataset = LaminetDataset('laminet_synthetic_dataset.json')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = LaminetMicro().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, field_energy = model(inputs)
            loss = loss_fn(outputs.unsqueeze(0), labels) + 0.01 * field_energy
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
