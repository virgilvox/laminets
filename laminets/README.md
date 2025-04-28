# Laminet Prototype

This directory contains the implementation of a minimal Laminet model prototype. Laminets are neural architectures that model sequential information as continuous semantic fields rather than discrete tokens.

## Overview

The implementation includes:

1. **SemanticField** - A neural module that represents information as field points in a continuous space
2. **Laminet** - A complete model that initializes, evolves, and decodes semantic fields
3. **Training and Visualization** - Scripts for training on synthetic data and visualizing field evolution

## Key Components

- **Field Initialization**: Maps input embeddings to initial field configurations
- **Field Evolution**: Implements the dynamics that evolve fields over time based on semantic forces
- **Field Decoding**: Transforms the evolved field back into output embeddings

## Getting Started

The notebook at `notebooks/laminet_prototype.ipynb` provides a complete walkthrough of:
- Model implementation
- Synthetic dataset generation
- Training process
- Evaluation methods
- Visualization of field evolution

## Requirements

See the `requirements.txt` file in the parent directory for dependencies.

## Citations

This implementation is based on the concept of modeling information as continuous semantic fields that evolve according to learned dynamics, as described in the main project README. 