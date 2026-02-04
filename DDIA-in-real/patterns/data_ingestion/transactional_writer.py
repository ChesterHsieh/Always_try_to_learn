"""
Pattern Name: Transactional Writer

Problem:
    Writes to the target dataset can be partial or inconsistent if the job
    fails in the middle of the write. Downstream readers may observe
    half-written partitions or tables.

Solution:
    Use a transactional writer that stages all changes and commits them
    atomically. Either all changes are visible, or none are.

Use Cases:
    - Writing daily batch data into a warehouse table
    - Ingesting data from an external API into a lakehouse
    - Updating derived tables in a multi-step pipeline

Reference: Data Engineering Design Patterns (2025), Transactional Writer pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence
import logging

from patterns.utils.simple_db import SimpleDB


logger = logging.getLogger(__name__)


class Sink(Protocol):
    """Abstract target for transactional writes.

    A Sink implementation is responsible for:
    - Starting a transactional context
    - Writing one or more batches within that context
    - Committing or rolling back the context
    """

    def begin(self) -> Any:
        """Start a transactional context and return a handle (e.g. transaction id)."""

    def write_batch(self, tx: Any, records: Sequence[dict]) -> None:
        """Write a batch of records within the given transactional context."""

    def commit(self, tx: Any) -> None:
        """Commit the transactional context, making all writes visible."""

    def rollback(self, tx: Any) -> None:
        """Rollback the transactional context, discarding staged writes."""


@dataclass
class SimpleDbSink:
    """SimpleDB-backed sink to demonstrate transactional writes.

    This sink uses the in-memory SimpleDB implementation from patterns/utils/simple_db.py
    to stage all writes in a transaction and commit them atomically.
    """

    db: SimpleDB
    table_name: str

    def begin(self) -> int:
        """Start a new SimpleDB transaction."""
        # Ensure table exists before starting the transaction
        self.db.create_table(self.table_name)
        tx_id = self.db.begin_transaction()
        logger.info("Started SimpleDB transaction %s for table '%s'", tx_id, self.table_name)
        return tx_id

    def write_batch(self, tx: int, records: Sequence[dict]) -> None:
        """Write a batch of records into the table within the transaction."""
        for idx, record in enumerate(records):
            key = f"record_{idx}"
            # We rely on SimpleDB's transactional insert
            success = self.db.insert(self.table_name, key, record, transaction_id=tx)
            if not success:
                raise RuntimeError(f"Failed to insert record {key} into {self.table_name}")
        logger.info("Staged %d records into table '%s' in transaction %s", len(records), self.table_name, tx)

    def commit(self, tx: int) -> None:
        """Commit the SimpleDB transaction."""
        success = self.db.commit_transaction(tx)
        if not success:
            raise RuntimeError(f"Failed to commit transaction {tx}")
        logger.info("Committed transaction %s for table '%s'", tx, self.table_name)

    def rollback(self, tx: int) -> None:
        """Rollback the SimpleDB transaction."""
        success = self.db.rollback_transaction(tx)
        if not success:
            raise RuntimeError(f"Failed to rollback transaction {tx}")
        logger.info("Rolled back transaction %s for table '%s'", tx, self.table_name)


class TransactionalWriter:
    """High-level transactional writer for data ingestion.

    This class coordinates the Extract → Transform → Load steps and ensures
    that the Load step is executed transactionally.

    Docstring (中文說明):
        TransactionalWriter 封裝了典型的 ETL 流程，特別強調「寫入端的原子性」：
        - 先啟動交易
        - 執行資料提取與轉換
        - 一次性寫入並提交
        - 若任一步驟失敗則回滾，避免部分寫入
    """

    def __init__(
        self,
        sink: Sink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Initialize a TransactionalWriter.

        Args:
            sink: Target sink implementing the transactional write protocol.
            extract_fn: Function that returns an iterable of raw records.
            transform_fn: Optional function that transforms the iterable of
                records. If None, records are written as-is.
            on_error: Optional callback invoked with the exception when the
                pipeline fails. Can be used for logging or metrics.
        """
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn
        self._on_error = on_error

    def run(self) -> int:
        """Run the full transactional ingestion pipeline.

        Returns:
            The number of records successfully written.

        Raises:
            Any exception raised by extract / transform / sink operations
            after attempting a rollback.
        """
        tx = None
        try:
            # 1. Begin transactional context
            tx = self._sink.begin()
            logger.info("TransactionalWriter: started transaction %s", tx)

            # 2. Extract
            raw_records = list(self._extract_fn())
            logger.info("TransactionalWriter: extracted %d records", len(raw_records))

            # 3. Transform
            if self._transform_fn is not None:
                transformed_records = list(self._transform_fn(raw_records))
                logger.info(
                    "TransactionalWriter: transformed %d records into %d records",
                    len(raw_records),
                    len(transformed_records),
                )
            else:
                transformed_records = raw_records

            # 4. Write as a single batch
            self._sink.write_batch(tx, transformed_records)

            # 5. Commit
            self._sink.commit(tx)
            logger.info("TransactionalWriter: committed transaction %s", tx)
            return len(transformed_records)

        except Exception as exc:  # noqa: BLE001 - pattern demo, we want to catch broadly
            logger.exception("TransactionalWriter: pipeline failed: %s", exc)
            if tx is not None:
                try:
                    self._sink.rollback(tx)
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.exception("TransactionalWriter: rollback failed: %s", rollback_exc)
            if self._on_error is not None:
                self._on_error(exc)
            # Re-raise to let callers / tests observe the failure
            raise


def demo_transactional_writer() -> None:
    """Run a small demo of the Transactional Writer pattern using SimpleDB.

    This demo simulates:
    - A list of raw events to be ingested
    - A transform step that filters out invalid events (amount <= 0)
    - A transactional write into a SimpleDB table
    """
    print("=" * 60)
    print("🧪 Transactional Writer Demo (SimpleDB)")
    print("=" * 60)

    db = SimpleDB()
    table_name = "payments"
    sink = SimpleDbSink(db=db, table_name=table_name)

    raw_events = [
        {"user_id": "u1", "amount": 100, "currency": "USD"},
        {"user_id": "u2", "amount": -50, "currency": "USD"},  # invalid
        {"user_id": "u3", "amount": 200, "currency": "EUR"},
    ]

    def extract() -> Iterable[dict]:
        # In a real pipeline this could read from files, APIs, or message queues
        return list(raw_events)

    def transform(events: Iterable[dict]) -> Iterable[dict]:
        # Filter out invalid payments and normalize currency code
        for event in events:
            if event["amount"] <= 0:
                # Skip invalid records
                continue
            yield {
                "user_id": event["user_id"],
                "amount": event["amount"],
                "currency": event["currency"].upper(),
            }

    writer = TransactionalWriter(
        sink=sink,
        extract_fn=extract,
        transform_fn=transform,
    )

    written = writer.run()
    print(f"\n✅ Transactional Writer ingested {written} valid records into '{table_name}'")
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
class SparkTransactionalSink:
    """Spark-based sink for transactional writes using Delta Lake or similar.

    This sink uses Spark's transactional capabilities (e.g., Delta Lake)
    to ensure atomic writes. Writes are staged in a temporary location
    and then atomically committed by moving/renaming the partition.
    """

    spark: SparkSession
    output_path: str
    format: str = "delta"  # or "parquet", "iceberg", etc.
    partition_by: Optional[list[str]] = None
    mode: str = "overwrite"  # or "append"

    def begin(self) -> str:
        """Start a transactional context and return a transaction ID."""
        import uuid
        tx_id = str(uuid.uuid4())
        logger.info("Started Spark transaction %s for path '%s'", tx_id, self.output_path)
        return tx_id

    def write_batch(self, tx: str, records: Sequence[dict]) -> None:
        """Write a batch of records within the transaction.

        In Spark, we write to a temporary location first, then commit
        by moving the data atomically.
        """
        if not records:
            logger.warning("No records to write in transaction %s", tx)
            return

        # Create DataFrame from records
        df = self.spark.createDataFrame(records)

        # Write to temporary location (transaction staging)
        temp_path = f"{self.output_path}_temp_{tx}"
        writer = df.write.format(self.format).mode("overwrite")

        if self.partition_by:
            writer = writer.partitionBy(*self.partition_by)

        writer.save(temp_path)
        logger.info("Staged %d records to temp path '%s' in transaction %s", len(records), temp_path, tx)

    def commit(self, tx: str) -> None:
        """Commit the transaction by atomically moving data from temp to final location."""
        import shutil
        import os
        from pathlib import Path

        temp_path = f"{self.output_path}_temp_{tx}"
        final_path = Path(self.output_path)

        # In a real implementation, this would use atomic file system operations
        # For demo purposes, we simulate atomic commit
        if os.path.exists(temp_path):
            if final_path.exists():
                if self.mode == "overwrite":
                    shutil.rmtree(str(final_path))
                elif self.mode == "append":
                    # For append mode, merge the data
                    temp_df = self.spark.read.format(self.format).load(temp_path)
                    existing_df = self.spark.read.format(self.format).load(str(final_path))
                    combined_df = existing_df.union(temp_df)
                    combined_df.write.format(self.format).mode("overwrite").save(str(final_path))
                    shutil.rmtree(temp_path)
                    logger.info("Committed transaction %s (append mode)", tx)
                    return

            shutil.move(temp_path, str(final_path))
            logger.info("Committed transaction %s for path '%s'", tx, self.output_path)
        else:
            raise RuntimeError(f"Transaction staging path '{temp_path}' does not exist")

    def rollback(self, tx: str) -> None:
        """Rollback the transaction by deleting the temporary staging area."""
        import shutil
        import os

        temp_path = f"{self.output_path}_temp_{tx}"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
            logger.info("Rolled back transaction %s, removed temp path '%s'", tx, temp_path)
        else:
            logger.warning("Transaction %s temp path '%s' does not exist, nothing to rollback", tx, temp_path)


class SparkTransactionalWriter:
    """Spark-based transactional writer for data ingestion.

    This class uses PySpark to perform transactional writes, typically
    using Delta Lake or similar transactional storage formats.

    Docstring (中文說明):
        SparkTransactionalWriter 使用 PySpark 實現事務寫入：
        - 使用 SparkSession 處理大規模資料
        - 支援 Delta Lake 等事務性儲存格式
        - 將資料寫入臨時位置後原子性提交
    """

    def __init__(
        self,
        spark: SparkSession,
        sink: SparkTransactionalSink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Initialize a SparkTransactionalWriter.

        Args:
            spark: SparkSession instance.
            sink: SparkTransactionalSink instance.
            extract_fn: Function that returns an iterable of raw records.
            transform_fn: Optional function that transforms records.
            on_error: Optional error callback.
        """
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is not available. Install with: pip install pyspark")

        self._spark = spark
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn
        self._on_error = on_error

    def run(self) -> int:
        """Run the Spark transactional ingestion pipeline."""
        tx = None
        try:
            tx = self._sink.begin()
            logger.info("SparkTransactionalWriter: started transaction %s", tx)

            raw_records = list(self._extract_fn())
            logger.info("SparkTransactionalWriter: extracted %d records", len(raw_records))

            if self._transform_fn is not None:
                transformed_records = list(self._transform_fn(raw_records))
                logger.info(
                    "SparkTransactionalWriter: transformed %d records into %d records",
                    len(raw_records),
                    len(transformed_records),
                )
            else:
                transformed_records = raw_records

            self._sink.write_batch(tx, transformed_records)
            self._sink.commit(tx)
            logger.info("SparkTransactionalWriter: committed transaction %s", tx)
            return len(transformed_records)

        except Exception as exc:  # noqa: BLE001
            logger.exception("SparkTransactionalWriter: pipeline failed: %s", exc)
            if tx is not None:
                try:
                    self._sink.rollback(tx)
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.exception("SparkTransactionalWriter: rollback failed: %s", rollback_exc)
            if self._on_error is not None:
                self._on_error(exc)
            raise


# ============================================================================
# PyFlink Implementation
# ============================================================================

try:
    from pyflink.table import StreamTableEnvironment
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False


@dataclass
class FlinkTransactionalSink:
    """Flink-based sink for transactional writes.

    This sink uses Flink's transactional capabilities, typically with
    Kafka transactional producer or JDBC sink with transactions.
    """

    table_env: StreamTableEnvironment
    table_name: str
    connector: str = "jdbc"  # or "kafka", "filesystem", etc.
    connector_config: dict[str, str] = None

    def __init__(
        self,
        table_env: StreamTableEnvironment,
        table_name: str,
        connector: str = "jdbc",
        connector_config: Optional[dict[str, str]] = None,
    ):
        self.table_env = table_env
        self.table_name = table_name
        self.connector = connector
        self.connector_config = connector_config or {}

    def begin(self) -> str:
        """Start a transactional context."""
        import uuid
        tx_id = str(uuid.uuid4())
        logger.info("Started Flink transaction %s for table '%s'", tx_id, self.table_name)
        return tx_id

    def write_batch(self, tx: str, records: Sequence[dict]) -> None:
        """Write a batch of records within the transaction."""
        if not records:
            logger.warning("No records to write in transaction %s", tx)
            return

        # Create temporary table for staging
        temp_table_name = f"{self.table_name}_temp_{tx}"

        # Create DataFrame-like structure from records
        # In Flink 2.0+, we'd typically use Table API or DataStream API
        # For demo, we create a temporary view using SQL
        # Note: PyFlink 2.0+ uses newer Table API, descriptors are deprecated
        # This is a simplified example - real implementation would use proper Flink APIs
        logger.info("Staged %d records to temp table '%s' in transaction %s", len(records), temp_table_name, tx)

    def commit(self, tx: str) -> None:
        """Commit the transaction."""
        temp_table_name = f"{self.table_name}_temp_{tx}"
        # In real implementation, would execute SQL to merge temp table into main table
        logger.info("Committed transaction %s for table '%s'", tx, self.table_name)

    def rollback(self, tx: str) -> None:
        """Rollback the transaction."""
        temp_table_name = f"{self.table_name}_temp_{tx}"
        # Drop temporary table
        logger.info("Rolled back transaction %s, dropped temp table '%s'", tx, temp_table_name)


class FlinkTransactionalWriter:
    """Flink-based transactional writer for data ingestion.

    This class uses PyFlink to perform transactional writes in streaming
    or batch mode.

    Docstring (中文說明):
        FlinkTransactionalWriter 使用 PyFlink 實現事務寫入：
        - 支援流式和批次處理
        - 使用 Flink 的事務性連接器
        - 確保 exactly-once 語義
    """

    def __init__(
        self,
        table_env: StreamTableEnvironment,
        sink: FlinkTransactionalSink,
        extract_fn: Callable[[], Iterable[dict]],
        transform_fn: Optional[Callable[[Iterable[dict]], Iterable[dict]]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Initialize a FlinkTransactionalWriter.

        Args:
            table_env: StreamTableEnvironment instance.
            sink: FlinkTransactionalSink instance.
            extract_fn: Function that returns an iterable of raw records.
            transform_fn: Optional function that transforms records.
            on_error: Optional error callback.
        """
        if not FLINK_AVAILABLE:
            raise ImportError("PyFlink is not available. Install with: pip install apache-flink")

        self._table_env = table_env
        self._sink = sink
        self._extract_fn = extract_fn
        self._transform_fn = transform_fn
        self._on_error = on_error

    def run(self) -> int:
        """Run the Flink transactional ingestion pipeline."""
        tx = None
        try:
            tx = self._sink.begin()
            logger.info("FlinkTransactionalWriter: started transaction %s", tx)

            raw_records = list(self._extract_fn())
            logger.info("FlinkTransactionalWriter: extracted %d records", len(raw_records))

            if self._transform_fn is not None:
                transformed_records = list(self._transform_fn(raw_records))
                logger.info(
                    "FlinkTransactionalWriter: transformed %d records into %d records",
                    len(raw_records),
                    len(transformed_records),
                )
            else:
                transformed_records = raw_records

            self._sink.write_batch(tx, transformed_records)
            self._sink.commit(tx)
            logger.info("FlinkTransactionalWriter: committed transaction %s", tx)
            return len(transformed_records)

        except Exception as exc:  # noqa: BLE001
            logger.exception("FlinkTransactionalWriter: pipeline failed: %s", exc)
            if tx is not None:
                try:
                    self._sink.rollback(tx)
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.exception("FlinkTransactionalWriter: rollback failed: %s", rollback_exc)
            if self._on_error is not None:
                self._on_error(exc)
            raise


