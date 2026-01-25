# AGENTS.md - RL Self-Play Framework

> Guidance for AI coding agents working in this repository.

## Project Overview

High-performance symmetric self-play reinforcement learning framework using Rust (environment) + Python (training).

**Architecture**: Rust vectorized environment (PyO3) → Python PPO training → Single neural network controls both players via symmetric observation transform.

**Game**: SimpleDuel - 12x12 tactical combat with 13 actions, terrain system, items, and resource management.

---

## Build & Run Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install Python dependencies
pip install maturin torch numpy

# Build Rust environment (REQUIRED after any src/lib.rs changes)
maturin develop --release
```

### Training
```bash
# Default training (10M steps)
python train.py

# Fast config (1M steps, for testing)
python train.py --config fast

# Long config (50M steps)
python train.py --config long

# Resume from checkpoint
python train.py --resume ppo_checkpoint.pth
```

### Watch Games
```bash
python watch.py                    # PPO vs PPO
python watch.py --p2 rule          # PPO vs Rule-based
python watch.py --model ppo_final.pth --delay 0.5
```

### Rust Development
```bash
maturin develop --release  # Build release
cargo check                # Check Rust code
cargo clippy               # Lint Rust code
cargo fmt                  # Format Rust code
```

### No Test Suite
Verify changes by:
1. `cargo check` for Rust changes
2. `python train.py --config fast` for quick training sanity check
3. `python watch.py` to visually verify game behavior

---

## Code Style Guidelines

### Python

**Imports** - Group in order: stdlib → third-party → local
```python
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from config import Config
from models.registry import get_model
```

**Type Hints** - Required for function signatures
**Naming**: Classes `PascalCase`, functions `snake_case`, constants `UPPER_SNAKE_CASE`
**Dataclasses** - Use for configuration
**Error Handling** - Minimal, fail fast with assertions

### Rust

**Naming**: Types `PascalCase`, functions `snake_case`, constants `UPPER_SNAKE_CASE`
**Error Handling** - Use `thiserror` for custom errors
**PyO3 Bindings** - Keep Python interface minimal

---

## Architecture Patterns

### Extensibility System

**Model Registry** (`models/registry.py`):
```python
from models.registry import register_model, get_model, list_models

@register_model("my_model")
class MyModel(nn.Module): ...

model = get_model("actor_critic", obs_dim=160, action_dim=13)
```

**Algorithm Registry** (`algorithms/registry.py`):
```python
from algorithms.registry import register_algorithm, get_algorithm

@register_algorithm("my_algo")
class MyAlgo(BaseAlgorithm): ...

algo = get_algorithm("ppo", config=config, obs_dim=160, action_dim=13)
```

### Rust Environment Traits

**GameEnv** - Standard interface (allocates per call):
```rust
pub trait GameEnv: Send + Sync + Clone {
    fn new() -> Self;
    fn reset(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn step(&mut self, a1: usize, a2: usize) -> StepResult;
    fn obs_dim() -> usize;
    fn action_dim() -> usize;
}
```

**GameEnvZeroCopy** - High-performance interface (zero allocation):
```rust
pub trait GameEnvZeroCopy: Send + Sync + Clone {
    fn reset_into(&mut self, obs_p1: &mut [f32], obs_p2: &mut [f32], 
                  mask_p1: &mut [f32], mask_p2: &mut [f32]);
    fn step_into(&mut self, a1: usize, a2: usize, 
                 obs_p1: &mut [f32], obs_p2: &mut [f32],
                 mask_p1: &mut [f32], mask_p2: &mut [f32]) -> (f32, f32, bool, GameInfo);
}
```

### Vectorized Environments

- `VectorizedEnv` - Standard implementation (backward compatible)
- `VectorizedEnvZeroCopy` - Optimized with pre-allocated buffers

### Performance Features

**AMP (Automatic Mixed Precision)**:
```python
config.use_amp = True  # Enable in config
```

**torch.compile()**:
```python
config.use_compile = True
config.compile_mode = "default"  # or "reduce-overhead", "max-autotune"
```

**Pre-allocated Buffers** (`buffers.py`):
```python
from buffers import RolloutBuffer, RolloutBufferConfig

buffer = RolloutBuffer(RolloutBufferConfig(
    num_steps=2048, num_envs=16, obs_dim=160, action_dim=13,
    use_pinned_memory=True
))
```

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `MAP_SIZE` | 12 | lib.rs, utils.py |
| `MAX_HP` | 4 | lib.rs, utils.py |
| `MAX_ENERGY` | 7 | lib.rs, utils.py |
| `MAX_AMMO` | 6 | lib.rs, utils.py |
| `MAX_SHIELD` | 2 | lib.rs, utils.py |
| `OBS_DIM` | 160 | lib.rs, model.py |
| `ACTION_DIM` | 13 | lib.rs, model.py |

---

## Common Tasks

### Adding a New Model
1. Create model class in `model.py`
2. Add `@register_model("name")` decorator
3. Use via `get_model("name", obs_dim=..., action_dim=...)`

### Adding a New Algorithm
1. Create class extending `BaseAlgorithm` in `algorithms/`
2. Add `@register_algorithm("name")` decorator
3. Use via `get_algorithm("name", config=..., obs_dim=..., action_dim=...)`

### Adding a New Game
1. Implement `GameEnv` and `GameEnvZeroCopy` traits in `src/lib.rs`
2. Create corresponding `VectorizedEnv` wrapper
3. Register in `#[pymodule]`

### Adding a New Action
1. Add to action constants in `src/lib.rs`
2. Implement in `SimpleDuel::step()`
3. Update action mask logic
4. Update `ACTION_NAMES` in `utils.py`
5. Rebuild: `maturin develop --release`

---

## File Reference

| File | Purpose |
|------|---------|
| `src/lib.rs` | Rust environment (SimpleDuel, VectorizedEnv, VectorizedEnvZeroCopy) |
| `model.py` | Neural networks (ActorCritic, DuelingActorCritic) |
| `models/registry.py` | Model registration and factory |
| `algorithms/ppo.py` | PPO algorithm with AMP/compile support |
| `algorithms/registry.py` | Algorithm registration and factory |
| `algorithms/base.py` | Abstract base class for algorithms |
| `buffers.py` | Pre-allocated RolloutBuffer |
| `agents/rule_based.py` | Heuristic opponent |
| `train.py` | Main training loop |
| `watch.py` | Live game visualization |
| `config.py` | Training hyperparameters |
| `utils.py` | Game state parsing, replay utilities |

---

## Debugging Tips

1. **Rust changes not taking effect**: Run `maturin develop --release`
2. **CUDA out of memory**: Reduce `num_envs` or `batch_size`, or enable AMP
3. **NaN in training**: Check action masking, ensure no division by zero
4. **Agent not learning**: Verify reward shaping, check entropy coefficient
5. **Model not found**: Ensure `@register_model` decorator is applied
