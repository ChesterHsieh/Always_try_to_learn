# Quick Start Guide

## Installation

1. **Install Xcode Command-Line Tools** (if not already installed):
   ```bash
   xcode-select --install
   ```

2. **Navigate to the JAX playground directory**:
   ```bash
   cd /Users/chester/Desktop/Always_try_to_learn/jax
   ```

3. **Install dependencies with uv**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

## Verify Installation

```bash
python verify_jax.py
```

You should see output confirming:
- JAX imported successfully
- JAX version: 0.4.38
- Metal device detected
- All tests passing

## Run Examples

### Basic Examples
```bash
python basic_example.py
```

This demonstrates:
- Basic array operations
- JIT compilation
- Automatic differentiation
- Vectorization with vmap
- Random number generation

### Matrix Operations
```bash
python matrix_example.py
```

This demonstrates:
- Matrix operations (addition, multiplication, transpose)
- Solving linear systems
- Eigenvalues and eigenvectors
- Matrix decompositions (QR, SVD)
- Gradient computation
- Performance benchmarking

## Quick Test

Try this in Python:

```python
import jax
import jax.numpy as jnp

# Check device
print(f"Devices: {jax.devices()}")

# Simple computation
x = jnp.arange(10)
y = x ** 2
print(f"Input: {x}")
print(f"Output: {y}")
```

## Environment Variables

For certain workloads, you may need these environment variables:

```bash
# Enable PJRT compatibility (already set in scripts)
ENABLE_PJRT_COMPATIBILITY=1 python your_script.py

# Disable JIT (for debugging or certain workloads)
JAX_DISABLE_JIT=1 python your_script.py
```

## Common Commands

```bash
# Install new package
uv add package-name

# Update dependencies
uv sync

# Start IPython with JAX
ipython

# Run tests (if you add any)
pytest
```

## Tips

1. **Always activate the virtual environment** before running scripts:
   ```bash
   source .venv/bin/activate
   ```

2. **Use float32 for best performance** on Metal:
   ```python
   x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
   ```

3. **Enable JIT compilation** for better performance:
   ```python
   from jax import jit
   
   @jit
   def fast_function(x):
       return x ** 2 + 2 * x + 1
   ```

4. **Check available devices**:
   ```python
   import jax
   print(jax.devices())  # Should show [METAL(id=0)]
   ```

## Troubleshooting

### "Metal device not found"
- Check macOS version: `sw_vers` (need Sonoma 14.4+)
- Verify Apple Silicon: `system_profiler SPDisplaysDataType`

### "Import Error"
- Ensure virtual environment is activated
- Run: `uv sync` to reinstall dependencies

### "Performance issues"
- Use `float32` instead of `float64`
- Enable JIT compilation
- Check Metal device is being used

## Next Steps

- Read the [main README](README.md) for detailed information
- Explore JAX documentation: https://jax.readthedocs.io/
- Check out Apple's JAX guide: https://developer.apple.com/metal/jax/
- Experiment with the example scripts and modify them
