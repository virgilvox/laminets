# Basic force functions

import torch

def semantic_attraction(point_i, point_j):
    direction = point_j.position - point_i.position
    distance = direction.norm(p=2) + 1e-6
    force_magnitude = (point_i.charge * point_j.charge) / (distance**2)
    return force_magnitude * direction / distance
