#!/usr/bin/env python3
"""
Matrix operations and benchmarking with JAX on Metal.
"""

import os
import time

# Enable PJRT compatibility for newer jaxlib versions
os.environ.setdefault('ENABLE_PJRT_COMPATIBILITY', '1')

import jax
import jax.numpy as jnp
from jax import jit


def print_section(title):
    """Helper to print section headers."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def basic_matrix_operations():
    """Demonstrate basic matrix operations."""
    print_section("Basic Matrix Operations")
    
    # Create matrices (use float32 for Metal compatibility)
    A = jnp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]], dtype=jnp.float32)
    
    B = jnp.array([[9.0, 8.0, 7.0],
                   [6.0, 5.0, 4.0],
                   [3.0, 2.0, 1.0]], dtype=jnp.float32)
    
    print(f"Matrix A:\n{A}\n")
    print(f"Matrix B:\n{B}\n")
    
    # Matrix operations
    print(f"A + B:\n{A + B}\n")
    print(f"A * B (element-wise):\n{A * B}\n")
    print(f"A @ B (matrix multiplication):\n{A @ B}\n")
    print(f"A.T (transpose):\n{A.T}\n")
    
    # Linear algebra operations
    print(f"det(A): {jnp.linalg.det(A)}")
    print(f"trace(A): {jnp.trace(A)}")


def matrix_multiplication_benchmark():
    """Benchmark matrix multiplication."""
    print_section("Matrix Multiplication Benchmark")
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")
        
        # Create random matrices
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)
        
        A = jax.random.normal(key1, (size, size))
        B = jax.random.normal(key2, (size, size))
        
        # Compile the matmul (first run)
        @jit
        def matmul(a, b):
            return jnp.dot(a, b)
        
        # Warm-up (compilation)
        _ = matmul(A, B).block_until_ready()
        
        # Benchmark
        num_runs = 10
        start = time.time()
        for _ in range(num_runs):
            result = matmul(A, B).block_until_ready()
        end = time.time()
        
        avg_time = (end - start) / num_runs
        flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
        gflops = (flops / avg_time) / 1e9
        
        print(f"  Average time: {avg_time*1000:.2f} ms")
        print(f"  Performance: {gflops:.2f} GFLOPS")


def solving_linear_systems():
    """Demonstrate solving linear systems."""
    print_section("Solving Linear Systems")
    
    print("Note: Advanced linear algebra operations like solve() may not be")
    print("fully supported on Metal backend. Using alternative approach.\n")
    
    # Simple inverse approach
    A = jnp.array([[3.0, 2.0, -1.0],
                   [2.0, -2.0, 4.0],
                   [-1.0, 0.5, -1.0]], dtype=jnp.float32)
    
    b = jnp.array([[1.0], [-2.0], [0.0]], dtype=jnp.float32)
    
    print(f"Solving Ax = b using matrix operations")
    print(f"A:\n{A}\n")
    print(f"b: {b.ravel()}\n")
    
    # Use least squares approach which is more widely supported
    try:
        x = jnp.linalg.lstsq(A, b)[0]
        print(f"Solution x: {x.ravel()}")
        
        # Verify
        verification = jnp.dot(A, x)
        print(f"Verification Ax: {verification.ravel()}")
        print(f"Close to b: {jnp.allclose(verification, b)}")
    except Exception as e:
        print(f"Linear solve not fully supported on Metal: {type(e).__name__}")
        print("This is a known limitation of the Metal backend.")


def eigenvalues_eigenvectors():
    """Compute eigenvalues and eigenvectors."""
    print_section("Eigenvalues and Eigenvectors")
    
    print("Note: Eigenvalue decomposition may not be fully supported on Metal.\n")
    
    # Symmetric matrix
    A = jnp.array([[4.0, 1.0, 2.0],
                   [1.0, 3.0, 1.0],
                   [2.0, 1.0, 5.0]], dtype=jnp.float32)
    
    print(f"Matrix A:\n{A}\n")
    
    try:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = jnp.linalg.eigh(A)
        
        print(f"Eigenvalues: {eigenvalues}")
        print(f"\nEigenvectors:\n{eigenvectors}\n")
        
        # Verify: Av = λv for first eigenvector
        v = eigenvectors[:, 0]
        λ = eigenvalues[0]
        
        Av = jnp.dot(A, v)
        λv = λ * v
        
        print(f"Verification for first eigenvector:")
        print(f"Av: {Av}")
        print(f"λv: {λv}")
        print(f"Close: {jnp.allclose(Av, λv)}")
    except Exception as e:
        print(f"Eigenvalue decomposition not fully supported on Metal: {type(e).__name__}")
        print("This is a known limitation of the Metal backend.")


def matrix_decompositions():
    """Demonstrate matrix decompositions."""
    print_section("Matrix Decompositions")
    
    print("Note: Some matrix decompositions may not be fully supported on Metal.\n")
    
    # Create a matrix
    A = jnp.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0],
                   [10.0, 11.0, 12.0]], dtype=jnp.float32)
    
    print(f"Matrix A (4x3):\n{A}\n")
    
    # Try QR decomposition
    try:
        Q, R = jnp.linalg.qr(A)
        print(f"QR Decomposition:")
        print(f"Q shape: {Q.shape}")
        print(f"R shape: {R.shape}")
        print(f"Reconstruction close: {jnp.allclose(jnp.dot(Q, R), A)}\n")
    except Exception as e:
        print(f"QR decomposition not supported: {type(e).__name__}\n")
    
    # Try SVD decomposition
    try:
        U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
        print(f"SVD Decomposition:")
        print(f"U shape: {U.shape}")
        print(f"S (singular values): {S}")
        print(f"Vt shape: {Vt.shape}")
        
        # Reconstruct
        reconstruction = U @ jnp.diag(S) @ Vt
        print(f"Reconstruction close: {jnp.allclose(reconstruction, A)}")
    except Exception as e:
        print(f"SVD decomposition not supported: {type(e).__name__}")
        print("These are known limitations of the Metal backend.")


def gradient_computation_example():
    """Example of computing gradients for matrix operations."""
    print_section("Gradient Computation for Matrix Operations")
    
    def loss_function(W, x):
        """Simple loss: ||Wx||^2"""
        y = jnp.dot(W, x)
        return jnp.sum(y ** 2)
    
    # Create parameters
    W = jnp.array([[1.0, 2.0],
                   [3.0, 4.0],
                   [5.0, 6.0]])
    
    x = jnp.array([1.0, 1.0])
    
    print(f"Weight matrix W:\n{W}\n")
    print(f"Input vector x: {x}\n")
    
    # Compute loss
    loss = loss_function(W, x)
    print(f"Loss: {loss}\n")
    
    # Compute gradient with respect to W
    from jax import grad
    grad_fn = grad(loss_function, argnums=0)
    dL_dW = grad_fn(W, x)
    
    print(f"Gradient dL/dW:\n{dL_dW}")


def main():
    """Run all matrix examples."""
    print("JAX Matrix Operations with Metal Acceleration")
    print("=" * 60)
    
    # Display device info
    devices = jax.devices()
    print(f"\nRunning on: {devices}")
    
    # Run examples
    basic_matrix_operations()
    solving_linear_systems()
    eigenvalues_eigenvectors()
    matrix_decompositions()
    gradient_computation_example()
    matrix_multiplication_benchmark()
    
    print("\n" + "="*60)
    print("  All Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
