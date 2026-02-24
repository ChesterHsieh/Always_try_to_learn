# JAX Playground with Metal Acceleration

A JAX development environment configured for Apple Silicon with Metal acceleration support.

> **Setup Status**: ✅ Fully configured and tested on Apple M2
> 
> See [SETUP_SUMMARY.md](SETUP_SUMMARY.md) for complete setup details and [QUICKSTART.md](QUICKSTART.md) for quick instructions.

## Requirements

- Mac computers with Apple silicon or AMD GPUs
- Python 3.9 or later
- macOS Sonoma 14.4+ (for jax-metal 0.1.0)
- Xcode command-line tools

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a concise getting started guide.

## Setup

### 1. Install Xcode Command-Line Tools

```bash
xcode-select --install
```

### 2. Install Dependencies with uv

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Verify Installation

```bash
# Quick verification
python -c 'import jax; print(jax.numpy.arange(10))'

# Run the verification script
python verify_jax.py
```

Expected output:
```
✓ JAX imported successfully
✓ JAX version: 0.4.38
✓ Available devices: [METAL(id=0)]
✓ Simple computation test: [0 1 2 3 4 5 6 7 8 9]
✓ JAX with Metal is working correctly!
```

## Compatibility Notes

### Current Configuration

This project is configured with:
- **JAX/jaxlib**: 0.4.38 (compatible with jax-metal 0.1.1)
- **jax-metal**: 0.1.1 (latest version)
- **Python**: 3.9+ (tested with 3.13.5)
- **macOS**: Sonoma 14.4+ required

### jax-metal Version Compatibility

| jax-metal | macOS                             | jaxlib               |
| --------- | --------------------------------- | -------------------- |
| 0.1.0     | Sonoma 14.4+                      | >=v0.4.26           |
| 0.0.7     | Sonoma 14.4+                      | >=v0.4.26           |
| 0.0.6     | Sonoma 14.4 Beta                  | >=v0.4.22, >v0.4.24 |
| 0.0.5     | Sonoma 14.2+                      | >=v0.4.20, >v0.4.22 |

**Note**: The project pins JAX/jaxlib to 0.4.x versions for stability with jax-metal. Newer JAX versions (0.5+, 0.9+) may not be compatible.

### Running with Newer jaxlib Versions

If you need to use a jaxlib version beyond the minimum, enable compatibility mode:

```bash
ENABLE_PJRT_COMPATIBILITY=1 python your_script.py
```

All example scripts in this project automatically set `ENABLE_PJRT_COMPATIBILITY=1` for maximum compatibility.

## Examples

### Basic Example

```python
import jax
import jax.numpy as jnp

# Create an array
x = jnp.arange(10)
print(f"Array: {x}")

# Perform computation
y = x ** 2
print(f"Squared: {y}")

# Check available devices
print(f"Devices: {jax.devices()}")
```

Run the included example:

```bash
python basic_example.py
```

### Matrix Operations Example

Run matrix multiplication benchmark:

```bash
python matrix_example.py
```

## Known Limitations

The Metal plug-in is experimental and has some limitations:

- **Unsupported data types**: `np.float64`, `np.complex64`, `np.complex128`
- Not all JAX tests pass with Metal backend
- Some operations may fall back to CPU

For tracked issues, see: https://github.com/google/jax/issues?q=is%3Aissue+is%3Aopen+metal

## Project Structure

```
jax/
├── README.md              # This file
├── pyproject.toml         # Project configuration
├── verify_jax.py          # Installation verification script
├── basic_example.py       # Simple JAX operations
└── matrix_example.py      # Matrix operations benchmark
```

## Development

### Adding New Dependencies

```bash
uv add package-name
```

### Running with Compatibility Mode

Some workloads may require disabling JIT or enabling compatibility:

```bash
# Disable JIT (for certain workloads)
JAX_DISABLE_JIT=1 python your_script.py

# Enable PJRT compatibility
ENABLE_PJRT_COMPATIBILITY=1 python your_script.py

# Both together
JAX_DISABLE_JIT=1 ENABLE_PJRT_COMPATIBILITY=1 python your_script.py
```

## Resources

- [Apple Developer - Accelerated JAX on Mac](https://developer.apple.com/metal/jax/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [Metal Plug-in Issues](https://github.com/google/jax/issues?q=is%3Aissue+is%3Aopen+metal)

## Troubleshooting

### Import Error

If you see import errors, ensure:
1. Virtual environment is activated
2. Dependencies are installed: `uv sync`
3. Xcode command-line tools are installed

### Metal Device Not Found

If JAX doesn't detect Metal devices:
1. Check macOS version: `sw_vers`
2. Verify Apple Silicon or AMD GPU: `system_profiler SPDisplaysDataType`
3. Update jax-metal: `uv add --upgrade jax-metal`

### Performance Issues

For best performance:
- Use `jnp.float32` instead of `jnp.float64`
- Enable JIT compilation (default)
- Profile with JAX profiling tools
