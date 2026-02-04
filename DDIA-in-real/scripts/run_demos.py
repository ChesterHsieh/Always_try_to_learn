"""
Unified Demo Runner
Execute different pattern demos from a single entry point
"""

import argparse


def run_transactional_writer_demo():
    """Run Transactional Writer ingestion pattern demo"""
    from patterns.data_ingestion.transactional_writer import demo_transactional_writer
    demo_transactional_writer()


def run_idempotent_writer_demo():
    """Run Idempotent Writer ingestion pattern demo"""
    from patterns.data_ingestion.idempotent_writer import demo_idempotent_writer
    demo_idempotent_writer()


def run_upsert_writer_demo():
    """Run Upsert Writer ingestion pattern demo"""
    from patterns.data_ingestion.upsert_writer import demo_upsert_writer
    demo_upsert_writer()


def run_append_only_writer_demo():
    """Run Append-Only Writer ingestion pattern demo"""
    from patterns.data_ingestion.append_only_writer import demo_append_only_writer
    demo_append_only_writer()


def run_change_data_capture_demo():
    """Run Change Data Capture (CDC) ingestion pattern demo"""
    from patterns.data_ingestion.change_data_capture import demo_change_data_capture
    demo_change_data_capture()


def main():
    """Main entry point for demo runner"""
    parser = argparse.ArgumentParser(
        description="Run Data Engineering Design Patterns demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demos.py ingestion-tx-writer     # Run Transactional Writer pattern demo
  python scripts/run_demos.py ingestion-idempotent     # Run Idempotent Writer pattern demo
  python scripts/run_demos.py ingestion-upsert         # Run Upsert Writer pattern demo
  python scripts/run_demos.py ingestion-append-only    # Run Append-Only Writer pattern demo
  python scripts/run_demos.py ingestion-cdc            # Run Change Data Capture pattern demo
  python scripts/run_demos.py all                      # Run all demos
        """
    )
    
    parser.add_argument(
        "demo",
        choices=[
            "ingestion-tx-writer",
            "ingestion-idempotent",
            "ingestion-upsert",
            "ingestion-append-only",
            "ingestion-cdc",
            "all",
        ],
        help="Which demo to run",
    )
    
    args = parser.parse_args()
    
    demos = {
        "ingestion-tx-writer": run_transactional_writer_demo,
        "ingestion-idempotent": run_idempotent_writer_demo,
        "ingestion-upsert": run_upsert_writer_demo,
        "ingestion-append-only": run_append_only_writer_demo,
        "ingestion-cdc": run_change_data_capture_demo,
    }
    
    if args.demo == "all":
        print("=" * 60)
        print("Running all pattern demos...")
        print("=" * 60)
        for name, demo_func in demos.items():
            print(f"\n{'='*60}")
            print(f"Running {name}...")
            print(f"{'='*60}")
            demo_func()
            print()
    else:
        demos[args.demo]()


if __name__ == "__main__":
    main()

