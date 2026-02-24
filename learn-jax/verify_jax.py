#!/usr/bin/env python3
"""
Verification script for JAX with Metal acceleration.
Run this script to verify that JAX is properly installed and configured.
"""

import os
import sys

# Enable PJRT compatibility for newer jaxlib versions
os.environ.setdefault('ENABLE_PJRT_COMPATIBILITY', '1')


def main():
    print("Verifying JAX installation with Metal acceleration...\n")
    
    # Test 1: Import JAX
    try:
        import jax
        import jax.numpy as jnp
        print("✓ JAX imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JAX: {e}")
        sys.exit(1)
    
    # Test 2: Check JAX version
    try:
        print(f"✓ JAX version: {jax.__version__}")
    except Exception as e:
        print(f"✗ Failed to get JAX version: {e}")
    
    # Test 3: Check available devices
    try:
        devices = jax.devices()
        print(f"✓ Available devices: {devices}")
        
        # Check if Metal device is available
        device_types = [str(d.platform) for d in devices]
        if any('metal' in dt.lower() for dt in device_types):
            print("✓ Metal acceleration is available")
        else:
            print("⚠ Warning: Metal device not detected. Available platforms:", device_types)
    except Exception as e:
        print(f"✗ Failed to get devices: {e}")
        sys.exit(1)
    
    # Test 4: Simple computation
    try:
        x = jnp.arange(10)
        print(f"✓ Simple computation test: {x}")
    except Exception as e:
        print(f"✗ Computation failed: {e}")
        sys.exit(1)
    
    # Test 5: Basic operations
    try:
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        c = a + b
        print(f"✓ Vector addition: {a} + {b} = {c}")
    except Exception as e:
        print(f"✗ Vector operation failed: {e}")
        sys.exit(1)
    
    # Test 6: Matrix multiplication
    try:
        m1 = jnp.ones((3, 3))
        m2 = jnp.eye(3)
        result = jnp.dot(m1, m2)
        print(f"✓ Matrix multiplication successful")
    except Exception as e:
        print(f"✗ Matrix operation failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ JAX with Metal is working correctly!")
    print("="*60)
    print("\nYou can now run the example scripts:")
    print("  python basic_example.py")
    print("  python matrix_example.py")


if __name__ == "__main__":
    main()
