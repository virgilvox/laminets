def field_coherence_loss(field):
    center_of_mass = torch.mean(torch.stack([p.position for p in field.points]), dim=0)
    spread = torch.mean(torch.stack([
        (p.position - center_of_mass).norm(p=2)
        for p in field.points
    ]))
    return spread  # We want this minimized
