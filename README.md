# 🛠️ Laminets (Lamina Networks)
> _Field Evolution Models for Emergent Reasoning and Unified Multi-Modal AI_

## Overview

**Laminets** are an experimental AI architecture based on **continuous semantic field evolution**, rather than discrete token-token attention.

Traditional transformer-based models (e.g., BERT, GPT) rely on pairwise attention operations between tokens, accumulating meaning by stacking many layers.  
This approach, while effective, exhibits several fundamental limitations:

- **Quadratic scaling** in input size (`O(n²)`)
- **Artificial memory** structures (positional encodings, caches)
- **Fragile multi-modal integration** (separate pipelines for text, images, etc.)
- **Low interpretability** (attention maps do not easily correlate to reasoning structures)

**Laminets** propose a different foundation:  
Instead of discrete attention, **inputs are embedded into a latent field of points** that **evolve dynamically** under **semantic forces** over continuous simulated time.

This evolution produces emergent semantic structures — clusters, chains, spirals — representing reasoning, memory, and abstraction without requiring stacking hundreds of discrete layers.

## Core Concepts

| Component | Description |
|:---|:---|
| **Field Points** | Each input (text token, image patch, audio slice) becomes a latent particle defined by `position`, `velocity`, `mass`, and `semantic charge`. |
| **Evolution Engine** | Points evolve continuously under forces like semantic attraction, repulsion, temporal alignment, and entropy decay. |
| **Memory** | Stable topological features in the field (e.g., attractors, spirals) naturally encode persistent information. |
| **Reasoning** | Global semantic structures form from local interactions — reasoning emerges without discrete attention heads. |
| **Multi-Modal Fusion** | All modalities embed into the same field; no cross-attention or late fusion required. |

## Architecture Diagram

<img src="images/diagram.png" width="200">

## Comparison to Transformers

| Aspect | Transformers | Laminets |
|:---|:---|:---|
| Core mechanism | Token-token attention | Field evolution by semantic forces |
| Scaling | `O(n²)` (quadratic) | Approximately `O(n)` (depends on sparsity) |
| Reasoning | Emergent via deep stacking | Emergent via field resonance |
| Memory | Positional encodings, caches | Stable field attractors |
| Multi-modal handling | Separate encoder/decoder paths | Native fusion into common field |
| Interpretability | Weak (attention maps) | Field topology visualization |

## Why Field Evolution?

- **Continuous-Time Representation:**  
  Meaning structures are not artificially segmented into discrete steps but flow over continuous simulated time.

- **Emergent Reasoning Without Explicit Layers:**  
  Reasoning arises naturally from the dynamics of interacting forces, reducing the need for hundreds of stacked transformer layers.

- **Stable Memory Without Positional Hacks:**  
  Persistent field configurations act as organic memory structures.

- **Native Multi-Modal Embedding:**  
  Field points can originate from any modality; interaction is based on meaning proximity, not engineered adapters.

## Current Status

This repository provides:

- Code to **create synthetic narrative datasets**.
- A minimal **prototype architecture** (`LaminetMicro`) demonstrating the field evolution mechanism.
- Tools to **train, evaluate**, and **visualize field evolution**.
- Initial experimental scripts for community exploration.

At this stage, we invite researchers and engineers to **experiment**, **replicate**, and **extend** Laminets.  
Performance benchmarks against transformers and large-scale experiments are still to come.

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- PyTorch >= 2.0
- Matplotlib
- Numpy

## Example: Laminet-Micro Forward Pass

```python
from laminet_micro.model import LaminetMicro

model = LaminetMicro(embed_dim=64, decoder_out_dim=6)
output, potential_energy = model(input_embeddings)
```

Field evolution visualization:

```python
from visualization.field_animation import animate_field_evolution

animate_field_evolution(model.last_field)
```

## Repository Structure

| Folder | Purpose |
|:---|:---|
| `laminet_micro/` | Core model and field classes |
| `datasets/` | Synthetic dataset scripts |
| `examples/` | Training, evaluation, visualization |
| `training/` | Utilities (loss functions, helpers) |
| `visualization/` | Field animation and static plotting |
| `LICENSE` | MIT License |
| `requirements.txt` | Dependencies |

## Vision

We hypothesize that **field evolution models** (FEMs) can eventually replace attention-based transformers in applications requiring:

- Emergent reasoning
- Long-term stable memory
- Unified multi-modal understanding
- Interpretability of internal structure

Laminets represent the first practical exploration of this hypothesis.

We invite experimentation, critique, and extension of this architecture from the broader AI research community.

## License

This project is licensed under the MIT License.
