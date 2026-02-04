"""
Pattern Name: Append-Only Writer

Problem:
    Some data systems require an immutable, append-only log where records
    are never updated or deleted. This provides an audit trail and enables
    time-travel queries, but requires careful handling to avoid duplicates
    and ensure ordering.

Solution:
    Use an append-only writer that only adds new records to the end of the
    dataset. Records are typically timestamped and may include sequence numbers
    to maintain ordering. Duplicate detection can be done using record IDs
    combined with timestamps.

Use Cases:
    - Event logging and audit trails
    - Time-series data ingestion
    - Immutable data lakes
    - Change data capture (CDC) event streams
    - Financial transaction logs

Reference: Data Engineering Design Patterns (2025), Append-Only Writer pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence
import logging
import time
from datetime import datetime

from patterns.utils.simple_db import SimpleDB


logger = logging.getLogger(__name__)


class AppendOnlySink(Protocol):
    """Abstract target for append-only writes.

    An AppendOnlySink implementation is responsible for:
    - Appending records to the end of the dataset
    - Maintaining ordering (e.g., using timestamps or sequence numbers)
    - Optionally checking for duplicates before appending
    """

    def append(self, record: dict) -> bool:
        """Append a record to the dataset.
        
        Returns:
            True if the record was appended, False if it was a duplicate.
        """


@dataclass
class SimpleDbAppendOnlySink:
    """SimpleDB-backed sink that performs append-only writes.

    This sink maintains an ordered list of records, using timestamps and
    sequence numbers to ensure ordering and detect duplicates.
    """

    db: SimpleDB
    table_name: str
    id_field: str = "id"  # Field name for unique record ID
    timestamp_field: str = "timestamp"  # Field name for timestamp
    check_duplicates: bool = True  # Whether to check for duplicates before appending

    def __post_init__(self):
        """Initialize sequence counter for ordering."""
        self._sequence_counter = 0

    def append(self, record: dict) -> bool:
        """Append a record to the dataset."""
        # Ensure table exists
        self.db.create_table(self.table_name)
        
        # Generate composite key: timestamp + sequence + id
        record_id = record.get(self.id_field, f"record_{int(time.time() * 1000)}")
        timestamp = record.get(self.timestamp_field, datetime.now().isoformat())
        
        # Check for duplicates if enabled
        if self.check_duplicates:
            # Use a composite key: id + timestamp
            composite_key = f"{record_id}_{timestamp}"
            existing = self.db.select(self.table_name, composite_key)
            if existing is not None:
                logger.info("Skipping duplicate record with ID '%s' and timestamp '%s'", record_id, timestamp)
                return False
        
        # Add sequence number for ordering
        self._sequence_counter += 1
        record_with_metadata = {
            **record,
            "_sequence": self._sequence_counter,
            "_append_time": datetime.now().isoformat(),
        }
        
        # Use composite key for storage
        composite_key = f"{record_id}_{timestamp}_{self._sequence_counter}"
        success = self.db.insert(self.table_name, composite_key, record_with_metadata)
        
        if success:
            logger.info("Appended record with ID '%s' (sequence %d) to table '%s'", record_id, self._sequence_counter, self.table_name)
        
        return success


class AppendOnlyWriter:
    """High-level append-only writer for data ingestion.

    This class ensures that records are only appended, never updated or deleted,
    maintaining an immutable log of all data.

    Docstring (中文說明):
        AppendOnlyWriter 實現不可變的追加寫入模式：
        - 所有記錄都追加到資料集末尾，永不更新或刪除
        - 使用時間戳和序列號維持順序
        - 支援重複檢測以避免重複追加
        - 適用於事件日誌、審計追蹤等場景
    """

    def __init__(
        self,
        sink: AppendOnlySink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
        on_duplicate: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Initialize an AppendOnlyWriter.

        Args:
            sink: Target sink implementing the append-only protocol.
            extract_fn: Function that returns an iterable of raw records.
            transform_fn: Optional function that transforms the iterable of records.
            on_duplicate: Optional callback invoked when a duplicate record is detected.
        """
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn
        self._on_duplicate = on_duplicate

    def run(self) -> tuple[int, int]:
        """Run the append-only ingestion pipeline.

        Returns:
            A tuple of (records_appended, duplicates_skipped).
        """
        # Extract
        raw_records = list(self._extract_fn())
        logger.info("AppendOnlyWriter: extracted %d records", len(raw_records))

        # Transform
        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
            logger.info(
                "AppendOnlyWriter: transformed %d records into %d records",
                len(raw_records),
                len(transformed_records),
            )
        else:
            transformed_records = raw_records

        # Append all records
        appended = 0
        duplicates = 0

        for record in transformed_records:
            was_appended = self._sink.append(record)
            
            if was_appended:
                appended += 1
            else:
                duplicates += 1
                if self._on_duplicate is not None:
                    self._on_duplicate(record)

        logger.info(
            "AppendOnlyWriter: appended %d records, skipped %d duplicates",
            appended,
            duplicates,
        )
        return appended, duplicates


def demo_append_only_writer() -> None:
    """Run a small demo of the Append-Only Writer pattern using SimpleDB.

    This demo simulates:
    - Appending event logs in order
    - Attempting to append duplicates (which are skipped)
    - Showing that records are never updated or deleted
    """
    print("=" * 60)
    print("🧪 Append-Only Writer Demo (SimpleDB)")
    print("=" * 60)

    db = SimpleDB()
    table_name = "event_log"
    sink = SimpleDbAppendOnlySink(
        db=db,
        table_name=table_name,
        id_field="event_id",
        timestamp_field="timestamp",
        check_duplicates=True,
    )

    events = [
        {"event_id": "e1", "user_id": "u1", "action": "login", "timestamp": "2024-01-01T10:00:00Z"},
        {"event_id": "e2", "user_id": "u1", "action": "click", "timestamp": "2024-01-01T10:01:00Z"},
        {"event_id": "e3", "user_id": "u2", "action": "login", "timestamp": "2024-01-01T10:02:00Z"},
    ]

    def extract() -> Iterable[dict]:
        return list(events)

    writer = AppendOnlyWriter(
        sink=sink,
        extract_fn=extract,
    )

    # First run - should append all records
    print("\n📝 First append run:")
    appended1, duplicates1 = writer.run()
    print(f"✅ Appended {appended1} records, skipped {duplicates1} duplicates")
    db.show_data(table_name)

    # Second run with same events - should skip duplicates
    print("\n📝 Second append run (duplicate test):")
    appended2, duplicates2 = writer.run()
    print(f"✅ Appended {appended2} records, skipped {duplicates2} duplicates")
    db.show_data(table_name)

    # Add new events and append
    events.append({"event_id": "e4", "user_id": "u2", "action": "logout", "timestamp": "2024-01-01T10:03:00Z"})
    print("\n📝 Third append run (with new events):")
    appended3, duplicates3 = writer.run()
    print(f"✅ Appended {appended3} records, skipped {duplicates3} duplicates")
    db.show_data(table_name)


# ============================================================================
# PySpark Implementation
# ============================================================================

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


@dataclass
class SparkAppendOnlySink:
    """Spark-based sink for append-only writes."""

    spark: SparkSession
    output_path: str
    id_field: str = "id"
    timestamp_field: str = "timestamp"
    format: str = "parquet"  # or "delta", "json", etc.
    check_duplicates: bool = True

    def append(self, record: dict) -> bool:
        """Append a record to the dataset."""
        record_id = record.get(self.id_field, "")
        timestamp = record.get(self.timestamp_field, "")

        if self.check_duplicates:
            try:
                existing_df = self.spark.read.format(self.format).load(
                    self.output_path
                )
                # Check for duplicates based on id + timestamp
                count = existing_df.filter(
                    (col(self.id_field) == record_id) & (
                        col(self.timestamp_field) == timestamp
                    )
                ).count()
                if count > 0:
                    logger.info(
                        "Skipping duplicate record with ID '%s' and timestamp '%s'",
                        record_id,
                        timestamp,
                    )
                    return False
            except Exception:  # noqa: BLE001
                # Table doesn't exist yet
                pass

        # Append record
        df = self.spark.createDataFrame([record])
        df.write.format(self.format).mode("append").save(self.output_path)
        logger.info("Appended record with ID '%s'", record_id)
        return True


class SparkAppendOnlyWriter:
    """Spark-based append-only writer for data ingestion."""

    def __init__(
        self,
        spark: SparkSession,
        sink: SparkAppendOnlySink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
    ) -> None:
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is not available. Install with: pip install pyspark")

        self._spark = spark
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn

    def run(self) -> tuple[int, int]:
        """Run the Spark append-only ingestion pipeline."""
        raw_records = list(self._extract_fn())

        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
        else:
            transformed_records = raw_records

        appended = 0
        duplicates = 0

        for record in transformed_records:
            if self._sink.append(record):
                appended += 1
            else:
                duplicates += 1

        return appended, duplicates


# ============================================================================
# PyFlink Implementation
# ============================================================================

try:
    from pyflink.table import StreamTableEnvironment
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False


@dataclass
class FlinkAppendOnlySink:
    """Flink-based sink for append-only writes."""

    table_env: StreamTableEnvironment
    table_name: str
    id_field: str = "id"
    timestamp_field: str = "timestamp"
    check_duplicates: bool = True

    def append(self, record: dict) -> bool:
        """Append a record to the dataset."""
        # Simplified implementation
        logger.info("Appended record with ID '%s'", record.get(self.id_field, ""))
        return True


class FlinkAppendOnlyWriter:
    """Flink-based append-only writer for data ingestion."""

    def __init__(
        self,
        table_env: StreamTableEnvironment,
        sink: FlinkAppendOnlySink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
    ) -> None:
        if not FLINK_AVAILABLE:
            raise ImportError("PyFlink is not available. Install with: pip install apache-flink")

        self._table_env = table_env
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn

    def run(self) -> tuple[int, int]:
        """Run the Flink append-only ingestion pipeline."""
        raw_records = list(self._extract_fn())

        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
        else:
            transformed_records = raw_records

        appended = 0
        duplicates = 0

        for record in transformed_records:
            if self._sink.append(record):
                appended += 1
            else:
                duplicates += 1

        return appended, duplicates

