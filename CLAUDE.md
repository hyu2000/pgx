# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pgx** is a JAX-native library providing GPU-accelerated parallel game simulators for reinforcement learning. It supports 20+ games including Chess, Go, Shogi, Backgammon, and various MinAtar environments. All environments are designed to be JIT-compilable and vectorizable for high-performance training on accelerators.

## Development Commands

### Setup
```bash
# Install development dependencies
make install-dev

# Install the package
make install
```

### Testing
```bash
# Run full test suite (uses pytest with 4 workers)
make test

# Run tests with coverage reporting
make test-with-codecov

# Run tests for a specific game
python3 -m pytest tests/test_go.py -v

# Run doctests
python3 -m pytest --doctest-modules pgx --ignore pgx/experimental
```

### Code Quality
```bash
# Format code
make format

# Check code style and types
make check

# Clean build artifacts
make clean
```

### Individual linting tools
```bash
# Format with black
black pgx

# Sort imports
isort pgx

# Type checking
mypy --config pyproject.toml pgx --ignore-missing-imports

# Linting
flake8 --config pyproject.toml --ignore E203,E501,W503,E704,E741 pgx
```

## Architecture Overview

### Core Components
- **`pgx/core.py`**: Central environment interface (`Env`, `State`) and environment registry
- **`pgx/_src/types.py`**: Type definitions (`Array`, `PRNGKey`)
- **`pgx/_src/struct.py`**: JAX pytree dataclass utilities
- **Individual game files**: Each game (e.g., `chess.py`, `go.py`) implements the `Env` interface

### Game Implementation Pattern
Each game environment follows this structure:
1. **State representation**: Game-specific state using JAX arrays
2. **Core methods**: `init()`, `step()`, `observe()`, `legal_action_mask()`, `is_terminal()`
3. **Visualization**: SVG rendering via corresponding `_src/dwg/` files
4. **API version**: Each game has a version (v0, v1, v2) tracking compatibility

### Key Directories
- **`pgx/`**: Main library code with individual game implementations
- **`pgx/_src/`**: Internal utilities (drawing, games, types, etc.)
- **`pgx/experimental/`**: Experimental features and utilities
- **`examples/`**: Training examples (AlphaZero, PPO)
- **`tests/`**: Comprehensive test suite for all games
- **`docs/`**: Documentation and game-specific guides

## Environment Usage Patterns
A uv venv is expected to be set up at project root.

### Basic Environment Usage
```python
import pgx
env = pgx.make("go_9x9")
state = env.init(key)
state = env.step(state, action)  # For deterministic games
state = env.step(state, action, key)  # For stochastic games (v2 API)
```

### Vectorized Execution
```python
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))
```

### Stochastic vs Deterministic Games
- **Deterministic**: Chess, Go, Tic-tac-toe (no `key` needed for step)
- **Stochastic**: 2048, Backgammon, MinAtar games (require `key` for step in API v2)

## Testing Guidelines

We use pytest for testing.

- All tests must pass before submitting changes

## AlphaZero Example

The `examples/alphazero/` directory contains a complete AlphaZero implementation:
- **`train.py`**: Main training script with self-play and learning (Equinox-based)
- **`network.py`**: Neural network architectures (Equinox-based, migrated from Haiku)
- **`config.py`**: Hyperparameter configuration
- **Usage**: `python3 train.py env_id=go_9x9 seed=0`
- **Migration**: Uses Equinox instead of Haiku for better debugging and modern JAX patterns

## Important Implementation Details

### API Versioning
Each environment has a version (accessible via `env.version`). API v2 introduced explicit randomness handling for stochastic environments.

### JAX Compatibility
All functions are designed to work with JAX transformations:
- Use `jnp` instead of `np`
- Ensure pure functions for JIT compilation
- Handle PRNGKey properly for randomness

### Performance Considerations
- Environments are optimized for GPU/TPU execution
- Some environments (Go, Chess) may perform better on GPU than TPU
- Vectorized execution is crucial for training performance

### Neural Network Framework Migration
The project has migrated from Haiku to Equinox for neural network implementations:
- **Equinox**: Modern JAX neural network library with better debugging and Pythonic interfaces
- **Migration completed**: AlphaZero example now uses Equinox (`network.py`)
- **Key differences**: Equinox uses explicit state management for batch normalization
- **Benefits**: Better debugging, more intuitive model definition, cleaner gradients