"""
Main entry point for Data Engineering Design Patterns project

This project focuses on implementing Data Engineering Design Patterns
from the book "Data Engineering Design Patterns" (2025).
"""


def main():
    print("=" * 60)
    print("Data Engineering Design Patterns")
    print("=" * 60)
    print("\nThis project implements Data Engineering Design Patterns.")
    print("Running Transactional Writer pattern demo...")
    print("=" * 60)
    print()
    
    # Import and run the Transactional Writer pattern demo
    from patterns.data_ingestion.transactional_writer import demo_transactional_writer
    demo_transactional_writer()
    
    print("\n" + "=" * 60)
    print("Note: To run specific pattern demos, use:")
    print("  uv run scripts/run_demos.py <demo_name>")
    print("=" * 60)


if __name__ == "__main__":
    main()
