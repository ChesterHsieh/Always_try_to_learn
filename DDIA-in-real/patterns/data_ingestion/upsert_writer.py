"""
Pattern Name: Upsert Writer

Problem:
    When ingesting data, you may need to either insert new records or update
    existing ones based on a unique key. Simple insert operations will fail
    on duplicates, and separate insert/update logic is error-prone.

Solution:
    Use an upsert writer that performs "insert or update" operations atomically.
    If a record with the given key exists, update it; otherwise, insert it.

Use Cases:
    - Syncing data from external systems where records may change over time
    - Maintaining dimension tables in a data warehouse
    - Updating user profiles or account information
    - Handling slowly changing dimensions (SCD Type 1)

Reference: Data Engineering Design Patterns (2025), Upsert Writer pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence
import logging

from patterns.utils.simple_db import SimpleDB


logger = logging.getLogger(__name__)


class UpsertSink(Protocol):
    """Abstract target for upsert operations.

    An UpsertSink implementation is responsible for:
    - Checking if a record exists (using a unique key)
    - Inserting new records or updating existing ones atomically
    - Optionally tracking insert vs update counts
    """

    def upsert(self, record_id: str, record: dict) -> tuple[bool, bool]:
        """Upsert a record (insert or update).
        
        Returns:
            A tuple of (was_inserted, was_updated). One will be True.
        """


@dataclass
class SimpleDbUpsertSink:
    """SimpleDB-backed sink that performs upsert operations.

    This sink inserts new records or updates existing ones based on a record ID.
    """

    db: SimpleDB
    table_name: str
    id_field: str = "id"  # Field name in the record that contains the unique ID

    def upsert(self, record_id: str, record: dict) -> tuple[bool, bool]:
        """Upsert a record (insert or update)."""
        # Ensure table exists
        self.db.create_table(self.table_name)
        
        # Check if record exists
        existing = self.db.select(self.table_name, record_id)
        is_update = existing is not None
        
        # Insert or update
        success = self.db.insert(self.table_name, record_id, record)
        
        if success:
            if is_update:
                logger.info("Updated record with ID '%s' in table '%s'", record_id, self.table_name)
                return False, True
            else:
                logger.info("Inserted new record with ID '%s' into table '%s'", record_id, self.table_name)
                return True, False
        else:
            raise RuntimeError(f"Failed to upsert record with ID '{record_id}'")


class UpsertWriter:
    """High-level upsert writer for data ingestion.

    This class performs insert-or-update operations, making it suitable for
    syncing data where records may change over time.

    Docstring (中文說明):
        UpsertWriter 執行「插入或更新」操作：
        - 如果記錄已存在（根據唯一鍵），則更新它
        - 如果記錄不存在，則插入新記錄
        - 適用於需要同步外部系統變更的場景
    """

    def __init__(
        self,
        sink: UpsertSink,
        extract_fn: Callable[[], Iterable[dict]],
        id_extractor: Optional[Callable[[dict], str]] = None,
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
        on_insert: Optional[Callable[[str, dict], None]] = None,
        on_update: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        """
        Initialize an UpsertWriter.

        Args:
            sink: Target sink implementing the upsert protocol.
            extract_fn: Function that returns an iterable of raw records.
            id_extractor: Optional function that extracts a unique ID from a record.
                If None, uses the 'id' field.
            transform_fn: Optional function that transforms the iterable of records.
            on_insert: Optional callback invoked when a new record is inserted.
            on_update: Optional callback invoked when an existing record is updated.
        """
        self._sink = sink
        self._extract_fn = extract_fn
        self._id_extractor = id_extractor or (lambda r: str(r.get("id", "")))
        self._transform_fn = transform_fn
        self._on_insert = on_insert
        self._on_update = on_update

    def run(self) -> tuple[int, int]:
        """Run the upsert ingestion pipeline.

        Returns:
            A tuple of (records_inserted, records_updated).
        """
        # Extract
        raw_records = list(self._extract_fn())
        logger.info("UpsertWriter: extracted %d records", len(raw_records))

        # Transform
        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
            logger.info(
                "UpsertWriter: transformed %d records into %d records",
                len(raw_records),
                len(transformed_records),
            )
        else:
            transformed_records = raw_records

        # Upsert all records
        inserts = 0
        updates = 0

        for record in transformed_records:
            record_id = self._id_extractor(record)
            was_inserted, was_updated = self._sink.upsert(record_id, record)
            
            if was_inserted:
                inserts += 1
                if self._on_insert is not None:
                    self._on_insert(record_id, record)
            elif was_updated:
                updates += 1
                if self._on_update is not None:
                    self._on_update(record_id, record)

        logger.info(
            "UpsertWriter: inserted %d records, updated %d records",
            inserts,
            updates,
        )
        return inserts, updates


def demo_upsert_writer() -> None:
    """Run a small demo of the Upsert Writer pattern using SimpleDB.

    This demo simulates:
    - Initial user profile data
    - Updating existing profiles with new information
    - Adding new profiles
    """
    print("=" * 60)
    print("🧪 Upsert Writer Demo (SimpleDB)")
    print("=" * 60)

    db = SimpleDB()
    table_name = "user_profiles"
    sink = SimpleDbUpsertSink(db=db, table_name=table_name, id_field="user_id")

    # Initial user profiles
    initial_profiles = [
        {"user_id": "u1", "name": "Alice", "email": "alice@example.com", "age": 25},
        {"user_id": "u2", "name": "Bob", "email": "bob@example.com", "age": 30},
    ]

    def extract_initial() -> Iterable[dict]:
        return list(initial_profiles)

    def id_extractor(record: dict) -> str:
        return record["user_id"]

    writer = UpsertWriter(
        sink=sink,
        extract_fn=extract_initial,
        id_extractor=id_extractor,
    )

    # First run - should insert all records
    print("\n📝 First ingestion run (inserts):")
    inserts1, updates1 = writer.run()
    print(f"✅ Inserted {inserts1} records, updated {updates1} records")
    db.show_data(table_name)

    # Second run with updated data - should update existing and insert new
    updated_profiles = [
        {"user_id": "u1", "name": "Alice Smith", "email": "alice.smith@example.com", "age": 26},  # Updated
        {"user_id": "u2", "name": "Bob", "email": "bob@example.com", "age": 30},  # Same
        {"user_id": "u3", "name": "Charlie", "email": "charlie@example.com", "age": 28},  # New
    ]

    def extract_updated() -> Iterable[dict]:
        return list(updated_profiles)

    writer._extract_fn = extract_updated

    print("\n📝 Second ingestion run (updates + new inserts):")
    inserts2, updates2 = writer.run()
    print(f"✅ Inserted {inserts2} records, updated {updates2} records")
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
class SparkUpsertSink:
    """Spark-based sink for upsert operations using Delta Lake MERGE."""

    spark: SparkSession
    output_path: str
    id_column: str = "id"
    format: str = "delta"

    def upsert(self, record_id: str, record: dict) -> tuple[bool, bool]:
        """Upsert a record using Delta Lake MERGE operation."""
        df = self.spark.createDataFrame([record])

        try:
            # Read existing table
            existing_df = self.spark.read.format(self.format).load(self.output_path)
            existing_count = existing_df.filter(col(self.id_column) == record_id).count()
            is_update = existing_count > 0

            # Use Delta Lake MERGE for upsert
            from delta.tables import DeltaTable
            delta_table = DeltaTable.forPath(self.spark, self.output_path)

            delta_table.alias("target").merge(
                df.alias("source"),
                f"target.{self.id_column} = source.{self.id_column}"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

            return (not is_update, is_update)
        except Exception:
            # Table doesn't exist, create it
            df.write.format(self.format).mode("overwrite").save(self.output_path)
            return True, False


class SparkUpsertWriter:
    """Spark-based upsert writer for data ingestion."""

    def __init__(
        self,
        spark: SparkSession,
        sink: SparkUpsertSink,
        extract_fn: Callable[[], Iterable[dict]],
        id_extractor: Optional[Callable[[dict], str]] = None,
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
    ) -> None:
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is not available. Install with: pip install pyspark")

        self._spark = spark
        self._sink = sink
        self._extract_fn = extract_fn
        self._id_extractor = id_extractor or (lambda r: str(r.get("id", "")))
        self._transform_fn = transform_fn

    def run(self) -> tuple[int, int]:
        """Run the Spark upsert ingestion pipeline."""
        raw_records = list(self._extract_fn())

        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
        else:
            transformed_records = raw_records

        inserts = 0
        updates = 0

        for record in transformed_records:
            record_id = self._id_extractor(record)
            was_inserted, was_updated = self._sink.upsert(record_id, record)
            if was_inserted:
                inserts += 1
            elif was_updated:
                updates += 1

        return inserts, updates


# ============================================================================
# PyFlink Implementation
# ============================================================================

try:
    from pyflink.table import StreamTableEnvironment
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False


@dataclass
class FlinkUpsertSink:
    """Flink-based sink for upsert operations."""

    table_env: StreamTableEnvironment
    table_name: str
    id_column: str = "id"

    def upsert(self, record_id: str, record: dict) -> tuple[bool, bool]:
        """Upsert a record using Flink SQL INSERT ... ON DUPLICATE KEY UPDATE."""
        # Simplified implementation - real Flink would use proper SQL
        logger.info("Upserted record with ID '%s'", record_id)
        return True, False


class FlinkUpsertWriter:
    """Flink-based upsert writer for data ingestion."""

    def __init__(
        self,
        table_env: StreamTableEnvironment,
        sink: FlinkUpsertSink,
        extract_fn: Callable[[], Iterable[dict]],
        id_extractor: Optional[Callable[[dict], str]] = None,
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
    ) -> None:
        if not FLINK_AVAILABLE:
            raise ImportError("PyFlink is not available. Install with: pip install apache-flink")

        self._table_env = table_env
        self._sink = sink
        self._extract_fn = extract_fn
        self._id_extractor = id_extractor or (lambda r: str(r.get("id", "")))
        self._transform_fn = transform_fn

    def run(self) -> tuple[int, int]:
        """Run the Flink upsert ingestion pipeline."""
        raw_records = list(self._extract_fn())

        if self._transform_fn is not None:
            transformed_records = list(self._transform_fn(raw_records))
        else:
            transformed_records = raw_records

        inserts = 0
        updates = 0

        for record in transformed_records:
            record_id = self._id_extractor(record)
            was_inserted, was_updated = self._sink.upsert(record_id, record)
            if was_inserted:
                inserts += 1
            elif was_updated:
                updates += 1

        return inserts, updates

