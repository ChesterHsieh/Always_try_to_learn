#!/usr/bin/env python3
"""
Basic JAX examples demonstrating core functionality with Metal acceleration.
"""

import os

# Enable PJRT compatibility for newer jaxlib versions
os.environ.setdefault('ENABLE_PJRT_COMPATIBILITY', '1')

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap


def print_section(title):
    """Helper to print section headers."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def basic_operations():
    """Demonstrate basic array operations."""
    print_section("Basic Array Operations")
    
    # Create arrays
    x = jnp.array([1, 2, 3, 4, 5])
    print(f"Array x: {x}")
    print(f"Type: {type(x)}, dtype: {x.dtype}")
    
    # Operations
    print(f"\nSquare: {x ** 2}")
    print(f"Sum: {jnp.sum(x)}")
    print(f"Mean: {jnp.mean(x)}")
    print(f"Max: {jnp.max(x)}")
    
    # Broadcasting
    y = jnp.array([[1], [2], [3]])
    z = jnp.array([10, 20, 30])
    print(f"\nBroadcasting:\n{y} +\n{z} =\n{y + z}")


def jit_compilation():
    """Demonstrate JIT compilation."""
    print_section("JIT Compilation")
    
    def slow_function(x):
        """A function that can be JIT compiled."""
        return x ** 2 + 2 * x + 1
    
    fast_function = jit(slow_function)
    
    x = jnp.arange(1000)
    
    # First call compiles
    print("First call (compiling)...")
    result1 = fast_function(x)
    print(f"Result shape: {result1.shape}")
    
    # Second call uses compiled version
    print("\nSecond call (using compiled version)...")
    result2 = fast_function(x)
    print(f"Results match: {jnp.allclose(result1, result2)}")


def automatic_differentiation():
    """Demonstrate automatic differentiation."""
    print_section("Automatic Differentiation")
    
    def f(x):
        """Function: f(x) = x^3 + 2x^2 + 3x + 4"""
        return x**3 + 2*x**2 + 3*x + 4
    
    # Derivative: f'(x) = 3x^2 + 4x + 3
    df = grad(f)
    
    x = 2.0
    print(f"f({x}) = {f(x)}")
    print(f"f'({x}) = {df(x)}")
    print(f"Expected f'({x}) = 3*{x}^2 + 4*{x} + 3 = {3*x**2 + 4*x + 3}")


def vectorization():
    """Demonstrate automatic vectorization with vmap."""
    print_section("Vectorization with vmap")
    
    def matrix_vector_product(matrix, vector):
        """Compute matrix-vector product."""
        return jnp.dot(matrix, vector)
    
    # Create batch of matrices and vectors
    matrices = jnp.ones((5, 3, 3))  # 5 matrices of shape (3, 3)
    vectors = jnp.ones((5, 3))       # 5 vectors of shape (3,)
    
    # Vectorized matrix-vector product
    vectorized_mvp = vmap(matrix_vector_product)
    results = vectorized_mvp(matrices, vectors)
    
    print(f"Input: {matrices.shape[0]} matrices of shape {matrices.shape[1:]}")
    print(f"Input: {vectors.shape[0]} vectors of shape {vectors.shape[1:]}")
    print(f"Output shape: {results.shape}")
    print(f"First result: {results[0]}")


def random_numbers():
    """Demonstrate JAX's approach to random numbers."""
    print_section("Random Number Generation")
    
    # JAX requires explicit PRNG keys
    key = jax.random.PRNGKey(42)
    print(f"Initial key: {key}")
    
    # Split key for multiple random operations
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Generate random numbers
    random_array = jax.random.normal(subkey1, (3, 3))
    print(f"\nRandom normal array:\n{random_array}")
    
    random_uniform = jax.random.uniform(subkey2, (5,))
    print(f"\nRandom uniform array: {random_uniform}")


def device_info():
    """Display device information."""
    print_section("Device Information")
    
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i}:")
        print(f"  Platform: {device.platform}")
        print(f"  Device kind: {device.device_kind}")
        print(f"  ID: {device.id}")


def main():
    """Run all examples."""
    print("JAX Basic Examples with Metal Acceleration")
    print("==========================================")
    
    device_info()
    basic_operations()
    jit_compilation()
    automatic_differentiation()
    vectorization()
    random_numbers()
    
    print_section("Examples Complete!")
    print("\nTry running: python matrix_example.py")


if __name__ == "__main__":
    main()
