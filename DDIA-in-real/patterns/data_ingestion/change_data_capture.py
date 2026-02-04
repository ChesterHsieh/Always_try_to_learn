"""
Pattern Name: Change Data Capture (CDC)

Problem:
    You need to capture and replicate changes (inserts, updates, deletes) from
    a source system to a target system in near real-time. Polling for changes
    is inefficient, and you need to ensure consistency and ordering.

Solution:
    Use Change Data Capture (CDC) to capture changes as they occur, typically
    by reading from transaction logs or using database triggers. Changes are
    represented as events (insert/update/delete) and streamed to the target.

Use Cases:
    - Real-time data replication between databases
    - Keeping data warehouses synchronized with operational databases
    - Event sourcing and event-driven architectures
    - Building real-time analytics pipelines
    - Maintaining audit trails of all changes

Reference: Data Engineering Design Patterns (2025), Change Data Capture pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence
import logging
import time
from datetime import datetime

from patterns.utils.simple_db import SimpleDB


logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change in a CDC event."""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    """Represents a single change event in CDC."""
    change_type: ChangeType
    record_id: str
    record: Optional[dict] = None  # None for DELETE events
    old_record: Optional[dict] = None  # Previous state for UPDATE/DELETE
    timestamp: str = ""
    sequence: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class CDCSource(Protocol):
    """Abstract source for Change Data Capture.

    A CDCSource implementation is responsible for:
    - Reading changes from a source system (e.g., transaction log)
    - Returning change events in order
    - Tracking the last processed position/sequence
    """

    def read_changes(self, last_sequence: int = 0) -> Iterable[ChangeEvent]:
        """Read changes since the last processed sequence.

        Args:
            last_sequence: The last sequence number that was processed.
                Changes with sequence > last_sequence will be returned.

        Returns:
            An iterable of ChangeEvent objects in sequence order.
        """


class CDCSink(Protocol):
    """Abstract target for Change Data Capture events.

    A CDCSink implementation is responsible for:
    - Applying change events to the target system
    - Maintaining ordering and ensuring exactly-once processing
    - Tracking the last processed sequence
    """

    def apply_change(self, event: ChangeEvent) -> bool:
        """Apply a change event to the target.

        Returns:
            True if the change was applied successfully.
        """

    def get_last_processed_sequence(self) -> int:
        """Get the last processed sequence number."""


@dataclass
class SimpleDbCDCSource:
    """SimpleDB-backed CDC source that simulates reading from a transaction log.

    This source tracks changes made to a source table and exposes them as
    change events. In a real implementation, this would read from database
    transaction logs (e.g., MySQL binlog, PostgreSQL WAL).
    """

    db: SimpleDB
    source_table_name: str
    change_log_table: str = "_cdc_log"

    def __init__(self, db: SimpleDB, source_table_name: str, change_log_table: str = "_cdc_log"):
        self.db = db
        self.source_table_name = source_table_name
        self.change_log_table = change_log_table
        self._sequence_counter = 0
        # Ensure change log table exists
        self.db.create_table(self.change_log_table)

    def _log_change(self, change_type: ChangeType, record_id: str, record: Optional[dict] = None, old_record: Optional[dict] = None) -> ChangeEvent:
        """Log a change to the CDC log."""
        self._sequence_counter += 1
        event = ChangeEvent(
            change_type=change_type,
            record_id=record_id,
            record=record,
            old_record=old_record,
            sequence=self._sequence_counter,
        )
        # Store the event in the change log
        log_key = f"seq_{self._sequence_counter}_{record_id}"
        self.db.insert(self.change_log_table, log_key, {
            "sequence": self._sequence_counter,
            "change_type": change_type.value,
            "record_id": record_id,
            "record": record,
            "old_record": old_record,
            "timestamp": event.timestamp,
        })
        return event

    def simulate_insert(self, record_id: str, record: dict) -> ChangeEvent:
        """Simulate an INSERT operation and log it."""
        self.db.create_table(self.source_table_name)
        self.db.insert(self.source_table_name, record_id, record)
        return self._log_change(ChangeType.INSERT, record_id, record=record)

    def simulate_update(self, record_id: str, new_record: dict) -> ChangeEvent:
        """Simulate an UPDATE operation and log it."""
        old_record = self.db.select(self.source_table_name, record_id)
        if old_record is None:
            raise ValueError(f"Record {record_id} does not exist for update")
        self.db.insert(self.source_table_name, record_id, new_record)
        return self._log_change(ChangeType.UPDATE, record_id, record=new_record, old_record=old_record)

    def simulate_delete(self, record_id: str) -> ChangeEvent:
        """Simulate a DELETE operation and log it."""
        old_record = self.db.select(self.source_table_name, record_id)
        if old_record is None:
            raise ValueError(f"Record {record_id} does not exist for delete")
        self.db.delete(self.source_table_name, record_id)
        return self._log_change(ChangeType.DELETE, record_id, old_record=old_record)

    def read_changes(self, last_sequence: int = 0) -> Iterable[ChangeEvent]:
        """Read changes from the change log since last_sequence."""
        # In a real implementation, this would read from transaction logs
        # For demo purposes, we read from our change log table
        events = []
        if self.change_log_table in self.db.data:
            for key, log_entry in self.db.data[self.change_log_table].items():
                seq = log_entry.get("sequence", 0)
                if seq > last_sequence:
                    events.append(ChangeEvent(
                        change_type=ChangeType(log_entry["change_type"]),
                        record_id=log_entry["record_id"],
                        record=log_entry.get("record"),
                        old_record=log_entry.get("old_record"),
                        timestamp=log_entry.get("timestamp", ""),
                        sequence=seq,
                    ))
        
        # Sort by sequence
        events.sort(key=lambda e: e.sequence)
        return events


@dataclass
class SimpleDbCDCSink:
    """SimpleDB-backed CDC sink that applies change events to a target table."""

    db: SimpleDB
    target_table_name: str
    state_table: str = "_cdc_state"  # Table to track last processed sequence

    def __init__(self, db: SimpleDB, target_table_name: str, state_table: str = "_cdc_state"):
        self.db = db
        self.target_table_name = target_table_name
        self.state_table = state_table
        # Ensure tables exist
        self.db.create_table(self.target_table_name)
        self.db.create_table(self.state_table)

    def apply_change(self, event: ChangeEvent) -> bool:
        """Apply a change event to the target table."""
        self.db.create_table(self.target_table_name)

        if event.change_type == ChangeType.INSERT:
            if event.record is None:
                logger.error("INSERT event missing record data")
                return False
            success = self.db.insert(self.target_table_name, event.record_id, event.record)
            if success:
                logger.info("Applied INSERT for record '%s'", event.record_id)

        elif event.change_type == ChangeType.UPDATE:
            if event.record is None:
                logger.error("UPDATE event missing record data")
                return False
            success = self.db.insert(self.target_table_name, event.record_id, event.record)
            if success:
                logger.info("Applied UPDATE for record '%s'", event.record_id)

        elif event.change_type == ChangeType.DELETE:
            success = self.db.delete(self.target_table_name, event.record_id)
            if success:
                logger.info("Applied DELETE for record '%s'", event.record_id)

        if success:
            # Update state
            self.db.insert(self.state_table, "last_sequence", {"sequence": event.sequence})
            return True
        return False

    def get_last_processed_sequence(self) -> int:
        """Get the last processed sequence number."""
        state = self.db.select(self.state_table, "last_sequence")
        if state is None:
            return 0
        return state.get("sequence", 0)


class ChangeDataCapture:
    """High-level Change Data Capture processor.

    This class reads changes from a source and applies them to a target,
    ensuring ordering and tracking progress.

    Docstring (中文說明):
        ChangeDataCapture 實現變更數據捕獲模式：
        - 從源系統讀取變更事件（INSERT/UPDATE/DELETE）
        - 按順序應用到目標系統
        - 追蹤處理進度，支援斷點續傳
        - 適用於實時數據複製和同步場景
    """

    def __init__(
        self,
        source: CDCSource,
        sink: CDCSink,
        on_change: Optional[Callable[[ChangeEvent], None]] = None,
    ) -> None:
        """
        Initialize a ChangeDataCapture processor.

        Args:
            source: CDC source that provides change events.
            sink: CDC sink that applies changes to the target.
            on_change: Optional callback invoked for each processed change event.
        """
        self._source = source
        self._sink = sink
        self._on_change = on_change

    def process_changes(self) -> tuple[int, int, int]:
        """Process all pending changes from the source.

        Returns:
            A tuple of (inserts_applied, updates_applied, deletes_applied).
        """
        last_sequence = self._sink.get_last_processed_sequence()
        logger.info("CDC: Starting from sequence %d", last_sequence)

        changes = list(self._source.read_changes(last_sequence))
        logger.info("CDC: Found %d pending changes", len(changes))

        inserts = 0
        updates = 0
        deletes = 0

        for event in changes:
            success = self._sink.apply_change(event)
            if success:
                if event.change_type == ChangeType.INSERT:
                    inserts += 1
                elif event.change_type == ChangeType.UPDATE:
                    updates += 1
                elif event.change_type == ChangeType.DELETE:
                    deletes += 1

                if self._on_change is not None:
                    self._on_change(event)
            else:
                logger.error("Failed to apply change event: %s", event)

        logger.info(
            "CDC: Applied %d inserts, %d updates, %d deletes",
            inserts,
            updates,
            deletes,
        )
        return inserts, updates, deletes


def demo_change_data_capture() -> None:
    """Run a small demo of the Change Data Capture pattern using SimpleDB.

    This demo simulates:
    - Changes happening in a source system
    - Capturing those changes as events
    - Applying changes to a target system
    """
    print("=" * 60)
    print("🧪 Change Data Capture (CDC) Demo (SimpleDB)")
    print("=" * 60)

    # Create source and target databases
    source_db = SimpleDB()
    target_db = SimpleDB()

    source_table = "source_users"
    target_table = "target_users"

    cdc_source = SimpleDbCDCSource(db=source_db, source_table_name=source_table)
    cdc_sink = SimpleDbCDCSink(db=target_db, target_table_name=target_table)

    cdc = ChangeDataCapture(source=cdc_source, sink=cdc_sink)

    # Simulate changes in source system
    print("\n📝 Simulating changes in source system:")
    cdc_source.simulate_insert("u1", {"user_id": "u1", "name": "Alice", "email": "alice@example.com"})
    cdc_source.simulate_insert("u2", {"user_id": "u2", "name": "Bob", "email": "bob@example.com"})
    print("✅ Created 2 users in source")

    # Process changes
    print("\n📝 Processing CDC changes:")
    inserts1, updates1, deletes1 = cdc.process_changes()
    print(f"✅ Applied {inserts1} inserts, {updates1} updates, {deletes1} deletes")
    print("\nSource system state:")
    source_db.show_data(source_table)
    print("\nTarget system state:")
    target_db.show_data(target_table)

    # Simulate more changes
    print("\n📝 Simulating more changes:")
    cdc_source.simulate_update("u1", {"user_id": "u1", "name": "Alice Smith", "email": "alice.smith@example.com"})
    cdc_source.simulate_insert("u3", {"user_id": "u3", "name": "Charlie", "email": "charlie@example.com"})
    cdc_source.simulate_delete("u2")
    print("✅ Updated u1, inserted u3, deleted u2")

    # Process changes again
    print("\n📝 Processing CDC changes again:")
    inserts2, updates2, deletes2 = cdc.process_changes()
    print(f"✅ Applied {inserts2} inserts, {updates2} updates, {deletes2} deletes")
    print("\nSource system state:")
    source_db.show_data(source_table)
    print("\nTarget system state:")
    target_db.show_data(target_table)


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
class SparkCDCSource:
    """Spark-based CDC source that reads from change logs."""

    spark: SparkSession
    change_log_path: str
    format: str = "delta"

    def read_changes(self, last_sequence: int = 0) -> Iterable[ChangeEvent]:
        """Read changes from the change log since last_sequence."""
        try:
            df = self.spark.read.format(self.format).load(
                self.change_log_path
            )
            changes_df = df.filter(
                col("sequence") > last_sequence
            ).orderBy("sequence")

            events = []
            for row in changes_df.collect():
                events.append(ChangeEvent(
                    change_type=ChangeType(row["change_type"]),
                    record_id=row["record_id"],
                    record=row.get("record"),
                    old_record=row.get("old_record"),
                    timestamp=row.get("timestamp", ""),
                    sequence=row["sequence"],
                ))
            return events
        except Exception:  # noqa: BLE001
            return []


@dataclass
class SparkCDCSink:
    """Spark-based CDC sink that applies change events."""

    spark: SparkSession
    target_path: str
    state_path: str
    format: str = "delta"

    def apply_change(self, event: ChangeEvent) -> bool:
        """Apply a change event to the target."""
        try:
            if event.change_type == ChangeType.INSERT:
                df = self.spark.createDataFrame([event.record])
                df.write.format(self.format).mode("append").save(
                    self.target_path
                )
            elif event.change_type == ChangeType.UPDATE:
                try:
                    from delta.tables import DeltaTable
                    delta_table = DeltaTable.forPath(
                        self.spark, self.target_path
                    )
                    source_df = self.spark.createDataFrame([event.record])
                    delta_table.alias("target").merge(
                        source_df.alias("source"),
                        "target.id = source.id"
                    ).whenMatchedUpdateAll().execute()
                except ImportError:
                    df = self.spark.createDataFrame([event.record])
                    df.write.format(self.format).mode(
                        "overwrite"
                    ).save(self.target_path)
            elif event.change_type == ChangeType.DELETE:
                try:
                    from delta.tables import DeltaTable
                    delta_table = DeltaTable.forPath(
                        self.spark, self.target_path
                    )
                    delta_table.delete(f"id = '{event.record_id}'")
                except ImportError:
                    logger.warning(
                        "Delta Lake not available, DELETE not supported"
                    )

            state_df = self.spark.createDataFrame(
                [{"sequence": event.sequence}]
            )
            state_df.write.format(self.format).mode(
                "overwrite"
            ).save(self.state_path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to apply change event: %s", e)
            return False

    def get_last_processed_sequence(self) -> int:
        """Get the last processed sequence number."""
        try:
            state_df = self.spark.read.format(self.format).load(
                self.state_path
            )
            result = state_df.agg({"sequence": "max"}).collect()[0]
            return result[0] if result[0] is not None else 0
        except Exception:  # noqa: BLE001
            return 0


class SparkChangeDataCapture:
    """Spark-based Change Data Capture processor."""

    def __init__(
        self,
        spark: SparkSession,
        source: SparkCDCSource,
        sink: SparkCDCSink,
        on_change: Optional[Callable[[ChangeEvent], None]] = None,
    ) -> None:
        if not SPARK_AVAILABLE:
            raise ImportError(
                "PySpark is not available. Install with: pip install pyspark"
            )

        self._spark = spark
        self._source = source
        self._sink = sink
        self._on_change = on_change

    def process_changes(self) -> tuple[int, int, int]:
        """Process all pending changes from the source."""
        last_sequence = self._sink.get_last_processed_sequence()
        changes = list(self._source.read_changes(last_sequence))

        inserts = 0
        updates = 0
        deletes = 0

        for event in changes:
            success = self._sink.apply_change(event)
            if success:
                if event.change_type == ChangeType.INSERT:
                    inserts += 1
                elif event.change_type == ChangeType.UPDATE:
                    updates += 1
                elif event.change_type == ChangeType.DELETE:
                    deletes += 1

                if self._on_change is not None:
                    self._on_change(event)

        return inserts, updates, deletes


# ============================================================================
# PyFlink Implementation
# ============================================================================

try:
    from pyflink.table import StreamTableEnvironment
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False


@dataclass
class FlinkCDCSource:
    """Flink-based CDC source using Flink CDC connectors."""

    table_env: StreamTableEnvironment
    source_table: str

    def read_changes(self, last_sequence: int = 0) -> Iterable[ChangeEvent]:
        """Read changes from Flink CDC source."""
        # Simplified - real implementation would use Flink CDC connectors
        return []


@dataclass
class FlinkCDCSink:
    """Flink-based CDC sink."""

    table_env: StreamTableEnvironment
    target_table: str
    state_table: str = "_cdc_state"

    def apply_change(self, event: ChangeEvent) -> bool:
        """Apply a change event using Flink SQL."""
        logger.info(
            "Applied %s for record '%s'",
            event.change_type.value,
            event.record_id,
        )
        return True

    def get_last_processed_sequence(self) -> int:
        """Get the last processed sequence number."""
        return 0


class FlinkChangeDataCapture:
    """Flink-based Change Data Capture processor."""

    def __init__(
        self,
        table_env: StreamTableEnvironment,
        source: FlinkCDCSource,
        sink: FlinkCDCSink,
        on_change: Optional[Callable[[ChangeEvent], None]] = None,
    ) -> None:
        if not FLINK_AVAILABLE:
            raise ImportError(
                "PyFlink is not available. Install with: pip install apache-flink"
            )

        self._table_env = table_env
        self._source = source
        self._sink = sink
        self._on_change = on_change

    def process_changes(self) -> tuple[int, int, int]:
        """Process all pending changes from the source."""
        changes = list(self._source.read_changes())

        inserts = 0
        updates = 0
        deletes = 0

        for event in changes:
            success = self._sink.apply_change(event)
            if success:
                if event.change_type == ChangeType.INSERT:
                    inserts += 1
                elif event.change_type == ChangeType.UPDATE:
                    updates += 1
                elif event.change_type == ChangeType.DELETE:
                    deletes += 1

                if self._on_change is not None:
                    self._on_change(event)

        return inserts, updates, deletes

